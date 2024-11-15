import argparse
import os

import torch
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.utils import make_grid


def add_noise(x, a, b):
    noise = torch.randn_like(x)
    return a * x + b * noise

def normalize(x):
    return 2 * x - 1

def denormalize(x):
    return (x + 1) / 2

def main(image='data/mandrill.png', a=0.5, b=0.1, num_steps=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img = Image.open(image)
    x = F.pil_to_tensor(img)
    x = x.to(torch.float32) / 255.
    x = x.to(device)
    x = normalize(x)

    results = [x.clone().cpu()]

    for i in range(num_steps):
        x = add_noise(x, a, b)
        results.append(x.clone().cpu())

    # Save the results
    results = torch.stack(results, dim=0)
    results = denormalize(results)
    results = results.clamp(0, 1)
    grid = make_grid(results, nrow=5)
    grid = F.to_pil_image(grid)

    os.makedirs('output', exist_ok=True)

    grid.save('output/part-a-very-simple-diffusion.jpg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Diffusion Process with Configurable Parameters")
    parser.add_argument('--image', type=str, default='data/mandrill.png', help="Path to the input image")
    parser.add_argument('--a', type=float, default=0.5, help="Scaling factor for the original image")
    parser.add_argument('--b', type=float, default=0.1, help="Scaling factor for the noise")
    parser.add_argument('--steps', type=int, default=4, help="Number of diffusion steps")

    args = parser.parse_args()
    main(args.image, args.a, args.b, args.steps)