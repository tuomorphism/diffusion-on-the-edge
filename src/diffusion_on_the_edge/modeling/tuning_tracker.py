# tracking.py
import math, time, csv, os
from dataclasses import dataclass, field
from typing import Dict, Optional, Iterable
import torch
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def _safe_mean(x: Iterable[float]) -> float:
    x = list(x)
    return float(sum(x) / max(1, len(x)))


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            total += float(g.pow(2).sum().item())
    return math.sqrt(total) if total > 0 else 0.0


def triangle_metrics_from_prescaled(x_prescaled: torch.Tensor) -> Dict[str, float]:
    """
    x_prescaled in [-1, 1]. We rescale to side lengths in [0,1],
    sort per-row so c >= b >= a, and compute triangle inequality margin a+b-c.
    """
    if x_prescaled.numel() == 0:
        return dict(valid_rate=0.0, avg_margin=0.0, min_margin=0.0,
                    mean_side=0.0, num=0)

    x = (x_prescaled + 1.0) / 2.0
    a, b, c = torch.sort(x, dim=1).values.unbind(dim=1)
    margin = a + b - c
    valid = (margin > 0)
    return dict(
        valid_rate=float(valid.float().mean().item()),
        avg_margin=float(margin.mean().item()),
        min_margin=float(margin.min().item()),
        mean_side=float(x.mean().item()),
        num=int(x.shape[0]),
    )


@dataclass
class EMA:
    beta: float = 0.98
    value: Dict[str, float] = field(default_factory=dict)

    def update(self, scalars: Dict[str, float]) -> Dict[str, float]:
        out = {}
        for k, v in scalars.items():
            if k not in self.value:
                self.value[k] = v
            else:
                self.value[k] = self.beta * self.value[k] + (1 - self.beta) * v
            out[k + "_ema"] = self.value[k]
        return out


class TuningTracker:
    """
    Tracks:
      - loss, score_loss, delta_loss, direction_loss (both raw and EMA)
      - delta/teacher geometry (norms, ratio, cosine)
      - grad norm, LR
      - sampler validity & triangle margins for each outer epoch

    Writes:
      - Console summaries
      - Optional TensorBoard (runs/<name>)
      - Optional CSV (runs/<name>/log.csv)
    """
    def __init__(self,
                 log_name: str = "triangle_tuning",
                 use_tensorboard: bool = True,
                 write_csv: bool = True):
        self.log_dir = os.path.join("runs", log_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.writer = None
        if use_tensorboard and SummaryWriter is not None:
            self.writer = SummaryWriter(self.log_dir)

        self.csv_path = os.path.join(self.log_dir, "log.csv") if write_csv else None
        if self.csv_path and not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "global_step", "outer_epoch", "inner_epoch",
                    "loss", "score_loss", "delta_loss", "direction_loss",
                    "loss_ema", "score_loss_ema", "delta_loss_ema", "direction_loss_ema",
                    "delta_norm", "teacher_norm", "delta_ratio", "cosine_neg_teacher_delta",
                    "grad_norm", "lr", "t_mean"
                ])

        self.ema = EMA(beta=0.98)
        self.global_step = 0
        self._outer_start_time = None

        # running accumulators for pretty end-of-epoch summaries
        self._acc = {}

    def _acc_add(self, kv: Dict[str, float]):
        for k, v in kv.items():
            self._acc.setdefault(k, []).append(float(v))

    def on_outer_epoch_start(self, epoch: int,
                             raw_samples: torch.Tensor,
                             validity_mask: torch.Tensor):
        self._outer_start_time = time.time()
        total = int(raw_samples.shape[0])
        num_valid = int(validity_mask.sum().item())
        valid_rate = num_valid / max(1, total)
        tri = triangle_metrics_from_prescaled(raw_samples[validity_mask])

        msg = (f"[outer {epoch}] generated {total} samples | "
               f"valid: {num_valid} ({valid_rate:.3f}) | "
               f"avg_margin: {tri['avg_margin']:.4f} | min_margin: {tri['min_margin']:.4f}")
        print(msg)

        if self.writer:
            self.writer.add_scalar("sampler/valid_rate", valid_rate, epoch)
            self.writer.add_scalar("sampler/avg_margin", tri["avg_margin"], epoch)
            self.writer.add_scalar("sampler/min_margin", tri["min_margin"], epoch)
            self.writer.add_scalar("sampler/mean_side", tri["mean_side"], epoch)
            self.writer.add_scalar("sampler/num_valid", num_valid, epoch)

    @torch.no_grad()
    def _cosine(self, a: torch.Tensor, b: torch.Tensor) -> float:
        # Mean cosine similarity across batch; be robust to zero norms.
        an = F.normalize(a, dim=1, eps=1e-12)
        bn = F.normalize(b, dim=1, eps=1e-12)
        return float((an * bn).sum(dim=1).mean().item())

    def log_train_step(self,
                       *,
                       outer_epoch: int,
                       inner_epoch: int,
                       student: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       batch_t: torch.Tensor,
                       teacher_pred: torch.Tensor,
                       student_pred: torch.Tensor,
                       loss_dict: Dict[str, torch.Tensor]):
        """
        loss_dict keys: 'loss', 'score_loss', 'delta_loss', 'direction_loss'
        """
        self.global_step += 1

        # geometry
        delta = student_pred - teacher_pred
        delta_norm = float(delta.norm(dim=1).mean().item())
        teacher_norm = float(teacher_pred.norm(dim=1).mean().item())
        delta_ratio = float((delta_norm / max(teacher_norm, 1e-12)))
        cosine_neg_teacher_delta = self._cosine(-teacher_pred, delta)

        # grads & lr (after backward, before/after step—up to you; this reads current grads)
        gnorm = _grad_norm(student)
        lr = float(optimizer.param_groups[0].get("lr", 0.0))
        t_mean = float(batch_t.mean().item())

        scalars = {
            "loss": float(loss_dict["loss"].item()),
            "score_loss": float(loss_dict["score_loss"].item()),
            "delta_loss": float(loss_dict["delta_loss"].item()),
            "direction_loss": float(loss_dict["direction_loss"].item()),
            "delta_norm": delta_norm,
            "teacher_norm": teacher_norm,
            "delta_ratio": delta_ratio,
            "cosine_neg_teacher_delta": cosine_neg_teacher_delta,
            "grad_norm": gnorm,
            "lr": lr,
            "t_mean": t_mean,
        }
        scalars.update(self.ema.update({
            k: scalars[k]
            for k in ("loss", "score_loss", "delta_loss", "direction_loss")
        }))

        # accumulate for inner/outer summaries
        self._acc_add(scalars)

        # TensorBoard
        if self.writer:
            for k, v in scalars.items():
                self.writer.add_scalar(f"train/{k}", v, self.global_step)

        # CSV
        if self.csv_path:
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    self.global_step, outer_epoch, inner_epoch,
                    scalars["loss"], scalars["score_loss"],
                    scalars["delta_loss"], scalars["direction_loss"],
                    scalars["loss_ema"], scalars["score_loss_ema"],
                    scalars["delta_loss_ema"], scalars["direction_loss_ema"],
                    scalars["delta_norm"], scalars["teacher_norm"], scalars["delta_ratio"],
                    scalars["cosine_neg_teacher_delta"], scalars["grad_norm"], scalars["lr"],
                    scalars["t_mean"]
                ])

        # brief console ping every ~256 steps
        if (self.global_step % 256) == 0:
            print(
                f"[step {self.global_step}] "
                f"loss {scalars['loss']:.4f} | score {scalars['score_loss']:.4f} | "
                f"Δ/teach {scalars['delta_ratio']:.3f} | cos(-T,Δ) {cosine_neg_teacher_delta:.3f} | "
                f"gnorm {gnorm:.2f}"
            )

    def on_inner_epoch_end(self, outer_epoch: int, inner_epoch: int):
        # summarize inner epoch
        means = {k: _safe_mean(v) for k, v in self._acc.items()}
        print(
            f"[outer {outer_epoch} | inner {inner_epoch}] "
            f"loss {_safe_mean(self._acc.get('loss', [0])):.4f} | "
            f"score {_safe_mean(self._acc.get('score_loss', [0])):.4f} | "
            f"Δ/teach {_safe_mean(self._acc.get('delta_ratio', [0])):.3f} | "
            f"cos(-T,Δ) {_safe_mean(self._acc.get('cosine_neg_teacher_delta', [0])):.3f}"
        )
        self._acc.clear()  # reset for next inner epoch

    def on_outer_epoch_end(self, epoch: int):
        dur = time.time() - (self._outer_start_time or time.time())
        if self.writer:
            self.writer.add_scalar("meta/outer_epoch_seconds", dur, epoch)
        print(f"[outer {epoch}] finished in {dur:.1f}s\n")

    def close(self):
        if self.writer:
            self.writer.close()
