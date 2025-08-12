from functools import reduce
import numpy as np

def _stochastic_integrator_improved_euler(initial_pos: np.ndarray, timestep_array: np.ndarray, f, g) -> list:
    delta_t = (timestep_array[-1] - timestep_array[0]) / timestep_array.shape[0]
    sqrt_delta_t = np.sqrt(delta_t)

    def _step(trajectory: list, current_t: float) -> list:
        x_prev, t_prev = trajectory[-1]

        noise = g(t_prev) * np.random.randn(*x_prev.shape)
        x_pred = x_prev + delta_t * f(x_prev, t_prev) + sqrt_delta_t * noise

        # Improve regular euler method by taking a predictive step and averaging them
        drift_predicted = f(x_pred, delta_t + t_prev)
        total_drift = 0.5 * (f(x_prev, t_prev) + drift_predicted)

        # Final output
        noise_next = g(current_t) * np.random.randn(*x_prev.shape)
        updated_trajectory = trajectory + [(x_prev + delta_t * total_drift + sqrt_delta_t * noise_next, current_t)]
        return updated_trajectory
    whole_trajectory = reduce(_step, timestep_array[1:], [(initial_pos, timestep_array[0])])
    return whole_trajectory

INTEGRATION_METHODS = {
    'euler': _stochastic_integrator_improved_euler
}

def generate_trajectory(initial_pos: np.ndarray, t: float, f, g, delta_t = 0.001, method = 'euler') -> list:
    """
    Generates a sample from stochastic process with form dX_t = f(X_t, t)dt + g(t)dW
    """
    integrator_function = INTEGRATION_METHODS.get(method)
    if integrator_function is None:
        raise ValueError('Invalid integration method "{method}".')
    
    timestep_num = int(t / delta_t)
    timestep_array = np.linspace(0, t, timestep_num)
    return integrator_function(initial_pos, timestep_array, f, g)

def generate_sample_ou(initial_pos: np.ndarray, t: float, lambda_coeff: float, sigma_coeff: float) -> np.ndarray:
    """
    Generates a sample from stochastic process with form dX_t = -lambda dt + sigma dW using the analytic expression for the posterior distribution.
    """

    d = initial_pos.shape[1]
    norm_mean = np.exp(-lambda_coeff * t)
    mean = norm_mean * initial_pos
    variance = sigma_coeff ** 2 / (2 * lambda_coeff) * (1 - norm_mean ** 2)
    return np.random.normal(loc = mean, scale = np.sqrt(variance), size = (d, 1))
    

def ou_sample_and_score(dataset: np.ndarray, lambda_coeff = 1.0, sigma= 1.0, t = 1.0, n=5):
    samples = []
    for _ in range(n):
        x0 = dataset[np.random.randint(0, len(dataset))]
        t = np.random.uniform(0, t)

        decay = np.exp(-lambda_coeff * t)
        variance = (sigma**2 / (2 * lambda_coeff)) * (1 - np.exp(-2 * lambda_coeff * t))
        std_dev = np.sqrt(variance)

        expected_val = decay * x0
        x_t = np.random.normal(loc = expected_val, scale = std_dev, size=x0.shape)
        score = -(x_t - decay * x0) / variance

        samples.append((x_t, t, score))

    return samples