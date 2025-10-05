import math, torch, torch.nn as nn

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, emb_dim: int = 64, max_freq: float = 1000.0):
        super().__init__()
        assert emb_dim % 2 == 0, "emb_dim must be even"
        half = emb_dim // 2
        self.register_buffer("freqs", torch.exp(torch.linspace(0, math.log(max_freq), steps=half)))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1: t = t.unsqueeze(-1)
        if t.size(-1) != 1: t = t.mean(dim=-1, keepdim=True)
        angles = t * self.freqs  # [B,half]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [B,emb_dim]

class SimpleScoreNet(nn.Module):
    def __init__(self, input_dimension: int, layer_count: int = 2, hidden_dim: int = 1024, time_emb_dim: int = 64):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        in_dim = input_dimension + time_emb_dim

        layers = []
        layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True) ]
        for _ in range(max(0, layer_count - 1)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True) ]
        out = nn.Linear(hidden_dim, input_dimension)

        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        
        layers += [out]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2: x = x.view(x.size(0), -1)
        temb = self.time_emb(t)
        return self.net(torch.cat([x, temb], dim=-1))
