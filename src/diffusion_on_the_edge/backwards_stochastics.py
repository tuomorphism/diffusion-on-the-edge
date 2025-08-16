import torch

@torch.no_grad()
def reverse_pc_sampler_torch(
    x_T,
    score_model,
    f_fn,
    g_fn,
    T=1.0,
    N=1000,
    snr=0.15,
    eps_corrector=None,
    device=None,
    dtype=None,
):
    """
    One-step Predictor (reverse EM) + one-step Corrector (Langevin) per time level.
    """
    device = device or x_T.device
    dtype = dtype or x_T.dtype
    x = x_T.to(device=device, dtype=dtype)
    dt_pos = T / N
    dt = -dt_pos
    t = torch.full((x.shape[0], 1), T, device=device, dtype=dtype)

    for _ in range(N):
        t_scalar = t[0, 0].item()
        g = torch.as_tensor(g_fn(t_scalar), device=device, dtype=dtype)
        drift = f_fn(x, t) - (g**2) * score_model(x, t)
        x = x + drift * dt + g * torch.sqrt(torch.tensor(abs(dt), device=device, dtype=dtype)) * torch.randn_like(x)
        s = score_model(x, t)
        if eps_corrector is None:
            eps = (snr * g) ** 2
        else:
            eps = torch.as_tensor(eps_corrector, device=device, dtype=dtype)
        x = x + eps * s + torch.sqrt(2.0 * eps) * torch.randn_like(x)

        t = t - dt_pos

    return x
