import torch
from torch.utils.data import IterableDataset

@torch.jit.script
def ou_mean_var(t: torch.Tensor, lam: float, sigma: float):
    decay = torch.exp(torch.tensor(-lam) * t)
    var = (sigma * sigma) / (2.0 * lam) * (1.0 - torch.exp(torch.tensor(-2.0 * lam) * t))
    return decay, var

def ou_sample_exact(x0: torch.Tensor, t: torch.Tensor, lam: float, sigma: float):
    device, dtype = x0.device, x0.dtype
    decay, var = ou_mean_var(t.to(device=device, dtype=dtype), lam, sigma)
    while decay.ndim < x0.ndim:
        decay = decay.unsqueeze(-1)
        var = var.unsqueeze(-1)
    noise = torch.randn_like(x0)
    return decay * x0 + torch.sqrt(var) * noise

class OUDiffusionDatasetVectorized(IterableDataset):
    def __init__(self, x0_pool: torch.Tensor, T_max, lam, sigma, batch_size, batches_per_epoch, device=None, dtype=torch.float32, seed=None):
        super().__init__()
        self.x0_pool = x0_pool.to(device or torch.device("cpu"), dtype=dtype)
        self.T_max = float(T_max)
        self.lam = float(lam)
        self.sigma = float(sigma)
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.base_seed = seed

    def __iter__(self):
        N = self.x0_pool.shape[0]
        for _ in range(self.batches_per_epoch):
            idx = torch.randint(0, N, (self.batch_size,))
            x0 = self.x0_pool[idx]
            t = torch.rand((self.batch_size,), dtype=self.dtype, device=self.device) * self.T_max
            decay, var = ou_mean_var(t, self.lam, self.sigma)
            decay = decay.unsqueeze(-1)
            var = var.unsqueeze(-1)
            mu_t = decay * x0
            xt = mu_t + torch.sqrt(var) * torch.randn_like(x0)
            score = -(xt - mu_t) / var
            yield {"t": t, "x0": x0, "xt": xt, "mean": mu_t, "var": var, "score": score}
