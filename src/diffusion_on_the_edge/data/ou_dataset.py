import torch
from torch.utils.data import IterableDataset
from diffusion_on_the_edge.processes.torch_process import TorchProcess, OUTorchParams

class OUDiffusionDatasetVectorized(IterableDataset):
    def __init__(self, process: TorchProcess, x0_pool: torch.Tensor, T_max, batch_size, batches_per_epoch, device=None, dtype=torch.float32, seed=None):
        super().__init__()
        self.process = process
        self.x0_pool = x0_pool.to(device or torch.device("cpu"), dtype=dtype)
        self.T_max = float(T_max)
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
            eps = 1E-5
            t = torch.rand((self.batch_size,), dtype=self.dtype, device=self.device) * (self.T_max - eps) + eps
            mean, std = self.process.transition_mean_std(x0, t)
            xt = mean + std * torch.randn_like(x0)
            score = -(xt - mean) / (std ** 2 + 1E-9)
            yield {"t": t, "x0": x0, "xt": xt, "mean": mean, "var": std ** 2, "score": score}
