import torch
import numpy as np
from functools import reduce

def _stochastic_integrator_improved_euler(initial_pos: np.ndarray, timestep_array: np.ndarray, f, g) -> list:
    delta_t = (timestep_array[-1] - timestep_array[0]) / timestep_array.shape[0]
    sqrt_delta_t = np.sqrt(delta_t)

    def _step(trajectory: list, current_t: float) -> list:
        x_prev, t_prev = trajectory[-1]

        noise = g(t_prev) * np.random.randn(*x_prev.shape)
        x_pred = x_prev + delta_t * f(x_prev, t_prev) + sqrt_delta_t * noise

        drift_predicted = f(x_pred, delta_t + t_prev)
        total_drift = 0.5 * (f(x_prev, t_prev) + drift_predicted)

        noise_next = g(current_t) * np.random.randn(*x_prev.shape)
        updated_trajectory = trajectory + [(x_prev + delta_t * total_drift + sqrt_delta_t * noise_next, current_t)]
        return updated_trajectory

    whole_trajectory = reduce(_step, timestep_array[1:], [(initial_pos, timestep_array[0])])
    return whole_trajectory

INTEGRATION_METHODS = {
    'euler': _stochastic_integrator_improved_euler
}

def generate_trajectory(initial_pos: np.ndarray, t: float, f, g, delta_t=0.001, method='euler') -> list:
    integrator_function = INTEGRATION_METHODS.get(method)
    if integrator_function is None:
        raise ValueError(f'Invalid integration method "{method}".')

    timestep_num = int(t / delta_t)
    timestep_array = np.linspace(0, t, timestep_num)
    return integrator_function(initial_pos, timestep_array, f, g)

def _get_sample_params(t: float, lambda_coeff: float, sigma_coeff: float) -> tuple[float, float]:
    decay = np.exp(-lambda_coeff * t)
    variance = (sigma_coeff**2 / (2 * lambda_coeff)) * (1 - np.exp(-2 * lambda_coeff * t))
    return decay, variance

def generate_sample_ou(initial_pos: np.ndarray, t: float, lambda_coeff: float, sigma_coeff: float) -> np.ndarray:
    """
    Exact OU sample:
    dX_t = -lambda * X_t dt + sigma dW_t
    """
    decay, variance = _get_sample_params(t, lambda_coeff, sigma_coeff)
    return decay * initial_pos + np.sqrt(variance) * np.random.randn(*initial_pos.shape)


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
