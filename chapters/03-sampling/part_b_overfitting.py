import argparse
import math
import os

import torch
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

class NoiseScheduler(torch.nn.Module):
    def __init__(self, steps=24, beta_start=1e-4, beta_end=0.6):
        super(NoiseScheduler, self).__init__()
        self.steps = steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        beta = torch.linspace(beta_start, beta_end, steps)
        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, 0)

        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)

    def add_noise(self, x0, t, noise):
        """
        Adds arbitrary noise to an image
        :param x0: initial image
        :param t: step number, 0 indexed (0 <= t < steps)
        :param noise: noise to add
        :return: image with noise at step t
        """
        alpha_bar = self.alpha_bar[t]
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise

    def sample_prev_step(self, xt, t, pred_noise):
        z = torch.randn_like(xt)
        z[t.expand_as(z) == 0] = 0

        mean = (1 / torch.sqrt(self.alpha[t])) * (xt - (self.beta[t] / torch.sqrt(1 - self.alpha_bar[t])) * pred_noise)
        var = ((1 - self.alpha_bar[t - 1])  / (1 - self.alpha_bar[t])) * self.beta[t]
        sigma = torch.sqrt(var)

        x = mean + sigma * z
        return x

def normalize(x):
    return 2 * x - 1

def denormalize(x):
    return (x + 1) / 2

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 3, 3, padding=1)
        self.linear = torch.nn.Linear(1, 16)

    def forward(self, x, t):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        t = t.to(torch.float32)
        t = self.linear(t)
        x = x + t.view(-1, 16, 1, 1)
        x = self.conv2(x)
        return x

def main(beta_start=1e-4, beta_end=0.6, num_steps=24):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise_scheduler = NoiseScheduler(steps=num_steps, beta_start=beta_start, beta_end=beta_end).to(device)
    model = Model().to(device)

    x0 = torch.zeros(1, 3, 8, 8).to(device)
    x0[:, :, 2:6, 2:6] = 1
    x0 = normalize(x0)
    img = F.to_pil_image(denormalize(x0[0])).resize((256, 256), Image.NEAREST)
    img.save("part-b-overfitting-target.png")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    iterations = 10000

    for it in range(iterations):
        optimizer.zero_grad()
        noise = torch.randn_like(x0, device=device)
        t = torch.randint(0, noise_scheduler.steps, (1,), device=device)
        x_t = noise_scheduler.add_noise(x0, t, noise)
        pred_noise = model(x_t, t)
        loss = criterion(pred_noise, noise)
        if it % 100 == 0:
            print(f"Iteration {it}, Loss {loss.item()}")
        loss.backward()
        optimizer.step()

    x = torch.randn(1, 3, 8, 8).to(device)
    for step in range(num_steps-1, -1, -1):
        t = torch.tensor(step, device=device).view(1,)
        pred_noise = model(x, t)
        x = noise_scheduler.sample_prev_step(x, t, pred_noise)

    x = denormalize(x).clamp(0, 1)
    img = F.to_pil_image(x[0]).resize((256, 256), Image.NEAREST)
    img.save("part-b-overfitting-output.png")
    print("Image saved as part-b-overfitting-output.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Diffusion Process with Configurable Parameters")
    parser.add_argument('--beta-start', type=float, default=1e-4, help="Starting beta value")
    parser.add_argument('--beta-end', type=float, default=0.6, help="Ending beta value")
    parser.add_argument('--steps', type=int, default=24, help="Number of diffusion steps")
    args = parser.parse_args()

    main(args.beta_start, args.beta_end, args.steps)
