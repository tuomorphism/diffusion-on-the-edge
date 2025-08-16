import math

import torch
import torch.nn as nn

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

