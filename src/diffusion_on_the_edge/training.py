
import torch
from dataclasses import dataclass
from tqdm import tqdm
from dataclasses import dataclass

from .model import SimpleScoreNet
@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1E-4
    weight_decay: float = 0.0
    device: str = "cpu" # by default, run on CPU

def train_scorenet(
    model: SimpleScoreNet,
    dataloader: torch.utils.data.DataLoader,
    cfg: TrainConfig = TrainConfig(),
):
    """
    Train the SimpleScoreNet using MSE between predicted and true score values.
    Assumes dataloader yields batches with 'x', 't', and 'score' values needed for forward pass and loss calculation.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        n_samples = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for batch in pbar:
            optimizer.zero_grad()

            x = batch['x0']
            t = batch['t']
            var = batch['var'].to(device)
            omega = torch.clamp(var, min=1e-6)
            omega = omega / (omega.mean() + 1e-12)

            true_score = batch['score']

            # Forward pass
            pred = model(x, t)

            loss_elementwise = (true_score - pred) ** 2
            loss_scaled = omega * loss_elementwise
            loss = loss_scaled.mean()

            # Backward
            loss.backward()
            optimizer.step()
            
            batch_size = x.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size
            pbar.set_postfix(loss=running_loss / max(1, n_samples))

        epoch_loss = running_loss / max(1, n_samples)
        history.append(epoch_loss)
        print(f"Epoch {epoch+1}: avg loss = {epoch_loss:.6f}")

    result = {"model": model, "losses": history}

    return result