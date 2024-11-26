import math

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm, trange

from torchmetrics.image.fid import FrechetInceptionDistance


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


def calculate_fid(model, noise_scheduler, num_samples=1024, batch_size=64):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # Set up Frechet Inception Distance (FID) Metric
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    fid_metric.set_dtype(torch.float64)  # Use double precision for numerical stability
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    batches = num_samples // batch_size

    print("Getting Inception features for real dataset...")
    for i, (images, _) in zip(trange(batches), data_loader):
        fid_metric.update(images.to(device), real=True)

    print(f"Generating {num_samples} images, in {batches} batches...")
    model.eval()
    for _ in trange(batches):
        x = torch.randn(batch_size, 3, 32, 32).to(device)
        for step in range(noise_scheduler.steps-1, -1, -1):
            with torch.no_grad():
                t = torch.tensor(step, device=device).expand(x.size(0),)
                pred_noise = model(x, t)
                x = noise_scheduler.sample_prev_step(x, t, pred_noise)

        x = (x + 1) / 2  # Scale to [0, 1]
        x = x.clamp(0, 1)  # Clamp to [0, 1]

        fid_metric.update(x, real=False)

    # Compute FID score
    fid_score = fid_metric.compute()
    print(f"FID Score: {fid_score.item()}")
    return fid_score