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

def main(image='data/mandrill.png', beta_start=1e-4, beta_end=0.6, num_steps=24):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img = Image.open(image)
    x0 = F.pil_to_tensor(img)
    x0 = x0.to(torch.float32) / 255.
    x0 = x0.to(device)
    x0 = normalize(x0)
    x0 = x0.unsqueeze(0)

    noise_scheduler = NoiseScheduler(steps=num_steps, beta_start=beta_start, beta_end=beta_end).to(device)

    model = Model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    loss_history = []
    iterations = 1000

    for it in range(iterations):
        optimizer.zero_grad()
        noise = torch.randn_like(x0, device=device)
        t = torch.randint(0, noise_scheduler.steps, (1,1), device=device)
        x_t = noise_scheduler.add_noise(x0, t, noise)
        pred_noise = model(x_t, t)
        loss = criterion(pred_noise, noise)
        loss_history.append(loss.item())
        if it % 100 == 0:
            print(f"Iteration {it}, Loss {loss.item()}")
        loss.backward()
        optimizer.step()

    # Plot the loss history
    plt.plot(loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.grid()
    plt.savefig('part-a-loss_history.png')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Diffusion Process with Configurable Parameters")
    parser.add_argument('--image', type=str, default='data/mandrill.png', help="Path to the input image")
    parser.add_argument('--beta-start', type=float, default=1e-4, help="Starting beta value")
    parser.add_argument('--beta-end', type=float, default=0.6, help="Ending beta value")
    parser.add_argument('--steps', type=int, default=24, help="Number of diffusion steps")
    args = parser.parse_args()

    main(args.image, args.beta_start, args.beta_end, args.steps)
