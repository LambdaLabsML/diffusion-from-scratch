import argparse
import math
import os

import torch
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.utils import make_grid

class NoiseScheduler:
    def __init__(self, steps=24, beta_start=1e-4, beta_end=0.6):
        super(NoiseScheduler, self).__init__()
        self.steps = steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta = torch.linspace(beta_start, beta_end, steps)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, 0)

    def add_noise(self, x0, t):
        """
        Adds arbitrary noise to an image
        :param x0: initial image
        :param t: step number, 0 indexed (0 <= t < steps)
        :return: image with noise at step t
        """
        alpha_bar = self.alpha_bar[t]
        noise = torch.randn_like(x0)
        return math.sqrt(1 - beta) * x + math.sqrt(beta) * noise

def normalize(x):
    return 2 * x - 1

def denormalize(x):
    return (x + 1) / 2

def main(image='data/mandrill.png', beta_start=1e-4, beta_end=0.6, num_steps=24):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img = Image.open(image)
    x = F.pil_to_tensor(img)
    x = x.to(torch.float32) / 255.
    x = x.to(device)
    x = normalize(x)

    results = [x.clone().cpu()]

    noise_scheduler = NoiseScheduler(steps=num_steps, beta_start=beta_start, beta_end=beta_end)

    for t in range(num_steps):
        x = noise_scheduler.add_noise(x, t)
        results.append(x.clone().cpu())

    # Save the results
    results = torch.stack(results, dim=0)
    results = denormalize(results)
    results = results.clamp(0, 1)
    grid = make_grid(results, nrow=5)
    grid = F.to_pil_image(grid)

    os.makedirs('output', exist_ok=True)

    grid.save('output/part-c-beta-schedule.jpg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Diffusion Process with Configurable Parameters")
    parser.add_argument('--image', type=str, default='data/mandrill.png', help="Path to the input image")
    parser.add_argument('--beta-start', type=float, default=1e-4, help="Starting beta value")
    parser.add_argument('--beta-end', type=float, default=0.6, help="Ending beta value")
    parser.add_argument('--steps', type=int, default=24, help="Number of diffusion steps")
    args = parser.parse_args()

    main(args.image, args.beta_start, args.beta_end, args.steps)
