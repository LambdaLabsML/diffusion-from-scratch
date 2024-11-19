import argparse
import math

import torch
import torchinfo
import torchvision
import torchvision.transforms.functional as F
from torchvision.utils import make_grid

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
        t = t.view(-1, 1, 1, 1)
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

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create a matrix to hold the positional encodings
        pe = torch.zeros(max_len, d_model)

        # Compute the positional encodings
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register pe as a buffer to avoid updating it during backpropagation
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Retrieve the positional encodings
        return self.pe[x]

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim):
        super(ConvBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.proj = torch.nn.Linear(embed_dim, out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x, embedding):
        x = self.conv1(x)
        emb_proj = self.proj(embedding).view(-1, x.size(1), 1, 1)
        x = torch.nn.functional.relu(x + emb_proj)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        return x

class Model(torch.nn.Module):
    def __init__(self, num_steps=1000, embed_dim=16):
        super(Model, self).__init__()

        self.embed = PositionalEncoding(embed_dim, num_steps)

        self.enc1 = ConvBlock(1, 16, embed_dim)
        self.enc2 = ConvBlock(16, 32, embed_dim)
        self.bottleneck = ConvBlock(32, 64, embed_dim)
        self.upconv2 = torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64, 32, embed_dim)
        self.upconv1 = torch.nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(32, 16, embed_dim)
        self.final = torch.nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x, t):
        emb = self.embed(t)

        enc1 = self.enc1(x, emb)
        enc2 = self.enc2(torch.nn.functional.max_pool2d(enc1, 2), emb)
        bottleneck = self.bottleneck(torch.nn.functional.max_pool2d(enc2, 2), emb)
        dec2 = self.dec2(torch.cat([enc2, self.upconv2(bottleneck)], 1), emb)
        dec1 = self.dec1(torch.cat([enc1, self.upconv1(dec2)], 1), emb)
        out = self.final(dec1)
        return out

def train(batch_size=128, epochs=10, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise_scheduler = NoiseScheduler(steps=1000, beta_start=1e-4, beta_end=0.02).to(device)
    model = Model().to(device)
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # Train
    for epoch in range(epochs):
        loss_epoch = 0
        n = 0
        for x, _ in data_loader:
            x = x.to(device)
            optimizer.zero_grad()
            noise = torch.randn_like(x, device=device)
            t = torch.randint(0, noise_scheduler.steps, (x.size(0),), device=device)
            x_t = noise_scheduler.add_noise(x, t, noise)
            pred_noise = model(x_t, t)
            loss = criterion(pred_noise, noise)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            n += x.size(0)
        loss_epoch /= n
        print(f"Epoch {epoch}, Loss {loss_epoch}")

    torch.save(model.state_dict(), 'model_part_c.pth')

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise_scheduler = NoiseScheduler(steps=1000, beta_start=1e-4, beta_end=0.02).to(device)
    model = Model().to(device)
    model.load_state_dict(torch.load('model_part_c.pth', weights_only=True))
    model.eval()

    x = torch.randn(64, 1, 28, 28).to(device)
    for step in range(noise_scheduler.steps-1, -1, -1):
        with torch.no_grad():
            t = torch.tensor(step, device=device).expand(x.size(0),)
            pred_noise = model(x, t)
            x = noise_scheduler.sample_prev_step(x, t, pred_noise)

    x = denormalize(x).clamp(0, 1)

    # Create an image grid
    grid = make_grid(x, nrow=8, padding=2)
    grid = F.to_pil_image(grid)
    grid.save("part-c-mnist-output.png")
    print("Image saved as part-c-mnist-output.png")

def eval(fid_samples=50000):
    from part_b_eval import calculate_fid, print_flops

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise_scheduler = NoiseScheduler(steps=1000, beta_start=1e-4, beta_end=0.02).to(device)
    model = Model().to(device)
    model.load_state_dict(torch.load('model_part_c.pth', weights_only=True))
    model.eval()

    x = torch.randn(1, 1, 28, 28, device=device)
    t = torch.tensor(0, device=device)
    input_data = (x, t)
    torchinfo.summary(model, input_data=input_data)
    print_flops(model, input_data)
    print("Calculating FID...")
    fid_value = calculate_fid(model, noise_scheduler, device=device, num_samples=fid_samples)
    print(f"FID: {fid_value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Diffusion Process with Configurable Parameters")
    parser.add_argument('command', choices=['train', 'test', 'eval'], help="Command to execute")
    parser.add_argument('--batch-size', type=int, default=128, help="Batch size")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--fid-samples', type=int, default=50000, help="Number of samples for FID calculation")
    args = parser.parse_args()

    if args.command == 'train':
        train(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)
    elif args.command == 'test':
        test()
    elif args.command == 'eval':
        eval(fid_samples=args.fid_samples)
