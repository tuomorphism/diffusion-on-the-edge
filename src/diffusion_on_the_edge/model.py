import math
from dataclasses import dataclass
from typing import Optional, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

class SinusoidalTimeEmbedding(nn.Module):
    """
    NeRF-style sinusoidal embedding for scalar timesteps t in [0,1] (or any range).
    Returns [B, emb_dim].
    """
    def __init__(self, emb_dim: int = 64, max_freq: float = 1000.0):
        super().__init__()
        assert emb_dim % 2 == 0, "emb_dim must be even"
        half = emb_dim // 2
        # Frequencies geometrically spaced in [1, max_freq]
        self.register_buffer(
            "freqs",
            torch.exp(torch.linspace(0, math.log(max_freq), steps=half))
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: shape [B] or [B,1] or [B, ...] (we'll reduce to [B,1]).
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if t.size(-1) != 1:
            # If someone expanded t to match x (e.g., [B, D]), just take a representative scalar per-item.
            t = t.mean(dim=-1, keepdim=True)

        # [B, 1] * [half] -> [B, half]
        angles = t * self.freqs  # broadcast
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb  # [B, emb_dim]


class SimpleScoreNet(nn.Module):
    """
    Simple MLP score network:
      - Concatenates input x with a sinusoidal time embedding of t.
      - Produces a score vector with the same dimension as x.
    """
    def __init__(
        self,
        input_dimension: int,
        layer_count: int = 2,
        hidden_dim: int = 512,
        time_emb_dim: int = 64,
    ):
        super().__init__()
        self.input_dimension = input_dimension
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)

        in_dim = input_dimension + time_emb_dim

        layers = []
        # input -> first hidden
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        # hidden stack
        for _ in range(max(0, layer_count - 1)):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        # final projection to score dim
        layers.append(nn.Linear(hidden_dim, input_dimension))

        self.net = nn.Sequential(*layers)

        # Optional: Xavier init for stability
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D]
        t: [B] or [B, 1] or [B, D] (if expanded); we reduce/encode to [B, time_emb_dim]
        """
        if x.dim() != 2:
            # Flatten everything but batch dim so we stay general.
            x = x.view(x.size(0), -1)

        temb = self.time_emb(t)  # [B, time_emb_dim]
        inp = torch.cat([x, temb], dim=-1)
        return self.net(inp)  # [B, D]



@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 0.0
    device: str = "cuda"
    grad_clip: Optional[float] = 1.0
    use_amp: bool = True
    ema_decay: Optional[float] = 0.999


def train_scorenet(
    model: SimpleScoreNet,
    dataloader: torch.utils.data.DataLoader,
    cfg: TrainConfig = TrainConfig(),
):
    """
    Train the SimpleScoreNet using MSE between predicted and true score values.
    Assumes dataloader yields (x, t, true_score) batches.
    - Proper epoch averaging
    - Mixed precision (optional)
    - Gradient clipping (optional)
    - EMA (optional)
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=cfg.use_amp)

    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay is not None else None

    history = {"loss": []}

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        n_samples = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for x, t, true_score in pbar:
            x = x.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)
            true_score = true_score.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=cfg.use_amp):
                # Forward
                pred = model(x, t)
                loss = F.mse_loss(pred, true_score)

            # Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            batch_size = x.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size
            pbar.set_postfix(loss=running_loss / max(1, n_samples))

        epoch_loss = running_loss / max(1, n_samples)
        history["loss"].append(epoch_loss)
        print(f"Epoch {epoch+1}: avg loss = {epoch_loss:.6f}")

    result = {"model": model, "losses": history["loss"]}

    return result
