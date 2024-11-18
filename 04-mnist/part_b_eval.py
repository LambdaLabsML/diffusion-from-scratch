import argparse
import math
import os
import sys

import torch
import torchvision
import torchmetrics
import torchinfo
from fvcore.nn import ActivationCountAnalysis, FlopCountAnalysis

from part_a_mnist import NoiseScheduler, denormalize, Model
from mnist_classifier import MnistFeatureExtractor

def print_flops(model, input_data):
    flops = FlopCountAnalysis(model, input_data)
    print('=' * 90)
    print(f'{"Module":<40}{"Operator":<25}{"FLOPs":<25}')
    print('=' * 90)
    d = {}
    for module, counter in flops.by_module_and_operator().items():
        if module == "":
            continue
        k = tuple(module.split('.'))
        if k[:-1] in d:
            del d[k[:-1]]
        d[k] = counter
    for k, counter in d.items():
        module = '.'.join(k)
        for operator, count in counter.items():
            print(f'{module:<40}{operator:<25}{count:,}')
    print('=' * 90)
    print(f'Total FLOPs: {flops.total():,}')
    print('=' * 90)

def calculate_fid(model, noise_scheduler, num_samples=5000, num_steps=1000, device=torch.device('cpu')):
    feature_extractor = MnistFeatureExtractor()
    feature_extractor.load_state_dict(torch.load('mnist_classifier.pth', weights_only=True))
    feature_extractor.eval()

    fid = torchmetrics.image.fid.FrechetInceptionDistance(feature=feature_extractor,
                                                          reset_real_features=False,
                                                          normalize=True,
                                                          input_img_size=(1, 28, 28))

    fid.to(device)
    fid.set_dtype(torch.float64)

    # Get real samples
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    seed = 42
    torch.manual_seed(seed)
    indices = torch.randperm(len(dataset))[:num_samples]
    reals = torch.stack([dataset[i][0] for i in indices])
    reals = reals.to(device)
    fid.update(reals, real=True)

    # Get generated samples
    model.eval()
    x = torch.randn(5000, 1, 28, 28).to(device)
    for step in range(num_steps-1, -1, -1):
        with torch.no_grad():
            t = torch.tensor(step, device=device).expand(x.size(0),)
            pred_noise = model(x, t)
            x = noise_scheduler.sample_prev_step(x, t, pred_noise)

    x = denormalize(x).clamp(0, 1)
    fid.update(x, real=False)

    return fid.compute()


def main(beta_start=1e-4, beta_end=0.02, num_steps=1000, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise_scheduler = NoiseScheduler(steps=num_steps, beta_start=beta_start, beta_end=beta_end).to(device)


    # Test
    model = Model().to(device)
    model.load_state_dict(torch.load('model_part_a.pth', weights_only=True))
    model.eval()
    x = torch.randn(1, 1, 28, 28, device=device)
    t = torch.tensor(0, device=device)
    input_data = (x, t)
    torchinfo.summary(model, input_data=input_data)
    print_flops(model, input_data)

    fid_value = calculate_fid(model, noise_scheduler, device=device, num_samples=5000)

    print(f"FID: {fid_value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Diffusion Process with Configurable Parameters")
    parser.add_argument('--beta-start', type=float, default=1e-4, help="Starting beta value")
    parser.add_argument('--beta-end', type=float, default=0.02, help="Ending beta value")
    parser.add_argument('--steps', type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument('--batch-size', type=int, default=128, help="Batch size")
    args = parser.parse_args()

    main(args.beta_start, args.beta_end, args.steps, args.batch_size)
