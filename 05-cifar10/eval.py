import math

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm, trange

from torchmetrics.image.fid import FrechetInceptionDistance


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