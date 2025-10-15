# tuning_model.py
import torch
import torch.nn as nn
import copy

class ResidualScore(nn.Module):
    """
    Wraps a frozen base score model with a small residual head Δs(x,t).
    new_score(x,t) = base_score(x,t) + Δs(x,t)
    """
    def __init__(self, base_score_model: nn.Module, d: int = 3, hidden: int = 128):
        super().__init__()
        self.base = copy.deepcopy(base_score_model).eval()
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.delta = nn.Sequential(
            nn.Linear(d + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x, t)
        delta_out = self.delta_only(x, t)
        return base_out + delta_out # Combining the base output with tuning delta

    def delta_only(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, t.view(-1,1)], dim=1)
        return self.delta(inp)


def build_teacher_student(pretrained_score_model: nn.Module, hidden_size = 128, d = 3, device=None):
    """
    Returns (teacher, student).
    - teacher: frozen copy (anchor/trust region)
    - student: trainable residual over the same base
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = ResidualScore(pretrained_score_model, d=d, hidden=hidden_size).to(device)
    student = ResidualScore(pretrained_score_model, d=d, hidden=hidden_size).to(device)

    student.load_state_dict(teacher.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher, student
