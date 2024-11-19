import argparse

import torch
import torchvision
import torchmetrics
import torchinfo
from fvcore.nn import FlopCountAnalysis
from tqdm import trange, tqdm

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

def calculate_fid(model, noise_scheduler, num_samples=50000, device=torch.device('cpu')):
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
    if num_samples < len(dataset):
        indices = torch.randperm(len(dataset))[:num_samples]
        reals = torch.stack([dataset[i][0] for i in indices])
    else:
        reals = torch.stack([dataset[i][0] for i in range(len(dataset))])

    for i in trange(0, len(reals), 2048, desc='FID: Real Samples'):
        imgs = reals[i:i+2048].to(device)
        fid.update(imgs, real=True)

    # Get generated samples
    model.eval()
    for i in trange(0, num_samples, 2048, desc='FID: Generated Samples'):
        n = min(2048, num_samples - i)
        x = torch.randn(n, 1, 28, 28).to(device)
        for step in range(noise_scheduler.steps-1, -1, -1):
            with torch.no_grad():
                t = torch.tensor(step, device=device).expand(x.size(0),)
                pred_noise = model(x, t)
                x = noise_scheduler.sample_prev_step(x, t, pred_noise)

        x = denormalize(x).clamp(0, 1)
        fid.update(x, real=False)

    return fid.compute()

def negative_log_likelihood(model, noise_scheduler, num_samples=50000, device=torch.device('cpu')):
    seed = 42
    torch.manual_seed(seed)
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=True)

    model.eval()
    nll = 0
    remaining = num_samples
    for x, _ in tqdm(dataloader, total=len(dataloader), desc='NLL'):
        n = min(remaining, x.size(0))
        x = x[:n].to(device)
        t = torch.randint(0, noise_scheduler.steps, (n,), device=device)
        noise = torch.randn_like(x, device=device)
        x_noise = noise_scheduler.add_noise(x, noise, t)
        pred_noise = model(x_noise, t)
        loss = torch.nn.functional.mse_loss(pred_noise, noise, reduction='none')
        nll += loss.sum().item()
        remaining -= n
    nll /= num_samples
    return nll


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise_scheduler = NoiseScheduler(steps=1000, beta_start=1e-4, beta_end=0.02).to(device)

    # Test
    model = Model().to(device)
    model.load_state_dict(torch.load('model_part_a.pth', weights_only=True))
    model.eval()

    x = torch.randn(1, 1, 28, 28, device=device)
    t = torch.tensor(0, device=device)
    input_data = (x, t)
    torchinfo.summary(model, input_data=input_data)
    print_flops(model, input_data)

    print("Calculating FID...")
    fid_value = calculate_fid(model, noise_scheduler, device=device, num_samples=50000)
    print(f"FID: {fid_value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Diffusion Process with Configurable Parameters")
    args = parser.parse_args()

    main()
