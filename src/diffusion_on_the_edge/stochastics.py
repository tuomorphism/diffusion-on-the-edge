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
