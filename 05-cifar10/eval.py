import math

import torch
from tqdm import tqdm, trange


def discretized_gaussian_log_likelihood(x0, mean, log_var):
    """
    Compute the discretized Gaussian log-likelihood for pixel intensities.
    :param x0: Ground truth image (scaled to [-1, 1])
    :param mean: Predicted mean (from reconstructed x0)
    :param log_var: Log variance for the distribution
    :return: Negative log-likelihood in bits/dim
    """
    scale = torch.exp(0.5 * log_var.view(-1, 1, 1, 1))
    lower_bound = x0 - 1 / 255.0
    upper_bound = x0 + 1 / 255.0

    cdf_upper = 0.5 * (1 + torch.erf((upper_bound - mean) / (scale * math.sqrt(2))))
    cdf_lower = 0.5 * (1 + torch.erf((lower_bound - mean) / (scale * math.sqrt(2))))

    probs = torch.clamp(cdf_upper - cdf_lower, min=1e-12)
    log_probs = torch.log(probs)

    nll = -log_probs.mean(dim=[1, 2, 3])  # Mean over spatial dimensions
    return nll / math.log(2)  # Convert to bits/dim

def compute_variational_bound(noise_scheduler, model, x0, max_t_samples=100):
    """
    Compute the variational bound for NLL evaluation.
    :param noise_scheduler: Instance of NoiseScheduler
    :param model: Diffusion model
    :param x0: Ground truth image
    :param timesteps: Total diffusion timesteps
    :return: Variational bound in bits/dim
    """
    device = x0.device
    timesteps = noise_scheduler.steps
    batch_size = x0.size(0)
    total_nll = 0.0

    if max_t_samples < timesteps:
        t_values = torch.randint(0, timesteps, (max_t_samples,))
    else:
        t_values = torch.arange(timesteps)


    for t in tqdm(t_values, position=1):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        noise = torch.randn_like(x0)

        # Add noise
        x_t = noise_scheduler.add_noise(x0, t_tensor, noise)

        # Predict noise with model
        pred_noise = model(x_t, t_tensor)

        # Reconstruct x0
        reconstructed_x0 = noise_scheduler.estimate_x0(x_t, t_tensor, pred_noise)

        # Compute NLL for timestep t
        log_var = torch.log(noise_scheduler.beta[t_tensor])  # Assuming beta is used as variance
        nll = discretized_gaussian_log_likelihood(x0, reconstructed_x0, log_var)
        total_nll += nll

    return total_nll.mean()


def eval_nll(model, test_loader, noise_scheduler, max_images=16384):
    """
    Evaluate NLL on the test dataset.
    :param model: Trained diffusion model
    :param test_loader: DataLoader for test data
    :param noise_scheduler: Instance of NoiseScheduler
    :param timesteps: Total diffusion timesteps
    :return: Average NLL in bits/dim
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    total_nll = 0.0
    num_samples = 0

    with torch.no_grad():
        for x0, _ in tqdm(test_loader, total=math.ceil(max_images / test_loader.batch_size), position=0):
            x0 = x0.to(device)
            batch_nll = compute_variational_bound(noise_scheduler, model, x0, max_t_samples=256)
            total_nll += batch_nll * x0.size(0)
            num_samples += x0.size(0)
            if num_samples >= max_images:
                break

    avg_nll = total_nll / num_samples
    print(f"Average NLL (bits/dim): {avg_nll.item()}")
    return avg_nll
