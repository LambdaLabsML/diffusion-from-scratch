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
        alpha_bar = self.alpha_bar[t].view(-1, 1, 1, 1)
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
    def __init__(self, num_steps=1000):
        super(Model, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 64, 5, padding=2)
        self.conv2 = torch.nn.Conv2d(64, 64, 5, padding=2)
        self.conv3 = torch.nn.Conv2d(64, 64, 5, padding=2)
        self.conv4 = torch.nn.Conv2d(64, 64, 5, padding=2)
        self.conv5 = torch.nn.Conv2d(64, 64, 5, padding=2)
        self.conv6 = torch.nn.Conv2d(64, 1, 5, padding=2)


        self.embed = torch.nn.Sequential(
            torch.nn.Embedding(num_steps, 64),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
        )

        self.proj1 = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
        )
        self.proj2 = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
        )
        self.proj3 = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
        )
        self.proj4 = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
        )
        self.proj5 = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
        )

    def forward(self, x, t):
        emb = self.embed(t)

        conv1 = self.conv1(x)
        x = torch.nn.functional.relu(conv1)
        x = x + self.proj1(emb).view(-1, 64, 1, 1)

        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = x + self.proj2(emb).view(-1, 64, 1, 1)

        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = x + self.proj3(emb).view(-1, 64, 1, 1)

        x = self.conv4(x)
        x = torch.nn.functional.relu(x)
        x = x + self.proj4(emb).view(-1, 64, 1, 1)

        x = self.conv5(x)
        x = torch.nn.functional.relu(x)
        x = x + self.proj5(emb).view(-1, 64, 1, 1)

        x = self.conv6(x)
        return x

def main(beta_start=1e-4, beta_end=0.02, num_steps=1000, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise_scheduler = NoiseScheduler(steps=num_steps, beta_start=beta_start, beta_end=beta_end).to(device)
    model = Model().to(device)

    N = batch_size
    x0 = torch.zeros(N, 1, 8, 8).to(device)
    x0[:, :, 2:6, 2:6] = 1
    x0 = normalize(x0)
    img = F.to_pil_image(denormalize(x0[0])).resize((256, 256), Image.NEAREST)
    img.save("part-c-overfitting-target.png")

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    criterion = torch.nn.MSELoss()

    iterations = 50000

    for it in range(iterations):
        optimizer.zero_grad()
        noise = torch.randn_like(x0, device=device)
        t = torch.randint(0, noise_scheduler.steps, (N,), device=device)
        x_t = noise_scheduler.add_noise(x0, t, noise)
        pred_noise = model(x_t, t)
        loss = criterion(pred_noise, noise)
        if it % 1000 == 0:
            print(f"Iteration {it}, Loss {loss.item()}")
        loss.backward()
        optimizer.step()

    x = torch.randn(1, 1, 8, 8).to(device)
    for step in range(num_steps-1, -1, -1):
        t = torch.tensor(step, device=device).view(1,)
        pred_noise = model(x, t)
        x = noise_scheduler.sample_prev_step(x, t, pred_noise)

    x = denormalize(x)
    img = F.to_pil_image(x[0]).resize((256, 256), Image.NEAREST)
    img.save("part-c-overfitting-output.png")
    print("Image saved as part-c-overfitting-output.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Diffusion Process with Configurable Parameters")
    parser.add_argument('--beta-start', type=float, default=1e-4, help="Starting beta value")
    parser.add_argument('--beta-end', type=float, default=0.02, help="Ending beta value")
    parser.add_argument('--steps', type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument('--batch-size', type=int, default=128, help="Batch size")
    args = parser.parse_args()

    main(args.beta_start, args.beta_end, args.steps, args.batch_size)
