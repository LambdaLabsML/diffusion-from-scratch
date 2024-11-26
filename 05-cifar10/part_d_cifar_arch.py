import argparse
import math
from abc import abstractmethod

import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.utils import make_grid

ACTIVATION_FUNCTIONS = {
    'relu': torch.nn.ReLU,
    'silu': torch.nn.SiLU,
    'leakyrelu': torch.nn.LeakyReLU,
    'gelu': torch.nn.GELU,
}

class NoiseScheduler(torch.nn.Module):
    def __init__(self, steps=1000, beta_start=1e-4, beta_end=0.02):
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

    def estimate_x0(self, xt, t, pred_noise):
        t = t.view(-1, 1, 1, 1)
        return (xt - torch.sqrt(1 - self.alpha_bar[t]) * pred_noise) / torch.sqrt(self.alpha_bar[t])

def normalize(x):
    return 2 * x - 1

def denormalize(x):
    return (x + 1) / 2

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=1000):
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

class TimestepBlock(torch.nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        raise NotImplementedError

class ResnetBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, embed_channels, activation_fn=torch.nn.SiLU):
        super(ResnetBlock, self).__init__()
        self.in_layers = torch.nn.Sequential(
            torch.nn.GroupNorm(32, in_channels),
            activation_fn(),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.emb_layers = torch.nn.Sequential(
            activation_fn(),
            torch.nn.Linear(embed_channels, out_channels)
        )
        self.out_layers = torch.nn.Sequential(
            torch.nn.GroupNorm(32, out_channels),
            activation_fn(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        if in_channels != out_channels:
            self.shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = torch.nn.Identity()

    def forward(self, x, embedding):
        _input = x
        x = self.in_layers(x)
        emb_out = self.emb_layers(embedding).view(-1, x.size(1), 1, 1)
        x = x + emb_out
        x = self.out_layers(x)
        return x + self.shortcut(_input)

class Upsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x

class Downsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class TimestepBlockSequential(torch.nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for module in self:
            if isinstance(module, TimestepBlock):
                x = module(x, emb)
            else:
                x = module(x)
        return x

class Model(torch.nn.Module):
    def __init__(self, image_channels=3, model_channels=32, activation_fn=torch.nn.SiLU):
        super(Model, self).__init__()

        embed_dim = model_channels * 4
        self.embed = torch.nn.Sequential(
            PositionalEncoding(model_channels),
            torch.nn.Linear(model_channels, embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(embed_dim, embed_dim),
        )

        self.input_blocks = torch.nn.ModuleList([
            torch.nn.Conv2d(image_channels, model_channels, kernel_size=3, padding=1),

            ResnetBlock(model_channels, model_channels, embed_dim, activation_fn),
            ResnetBlock(model_channels, model_channels, embed_dim, activation_fn),
            Downsample(model_channels, model_channels),

            ResnetBlock(model_channels, model_channels*2, embed_dim, activation_fn),
            ResnetBlock(model_channels*2, model_channels*2, embed_dim, activation_fn),
            Downsample(model_channels*2, model_channels*2),

            ResnetBlock(model_channels*2, model_channels*2, embed_dim, activation_fn),
            ResnetBlock(model_channels*2, model_channels*2, embed_dim, activation_fn),
        ])

        self.middle_block = TimestepBlockSequential(
            ResnetBlock(model_channels*2, model_channels*2, embed_dim, activation_fn),
            ResnetBlock(model_channels*2, model_channels*2, embed_dim, activation_fn),
        )

        self.output_blocks = torch.nn.ModuleList([
            ResnetBlock(model_channels*4, model_channels*2, embed_dim, activation_fn),
            ResnetBlock(model_channels*4, model_channels*2, embed_dim, activation_fn),
            TimestepBlockSequential(
                ResnetBlock(model_channels*4, model_channels*2, embed_dim, activation_fn),
                Upsample(model_channels*2, model_channels*2),
            ),

            ResnetBlock(model_channels*4, model_channels*2, embed_dim, activation_fn),
            ResnetBlock(model_channels*4, model_channels*2, embed_dim, activation_fn),
            TimestepBlockSequential(
                ResnetBlock(model_channels*3, model_channels*2, embed_dim, activation_fn),
                Upsample(model_channels*2, model_channels*2),
            ),

            ResnetBlock(model_channels*3, model_channels, embed_dim, activation_fn),
            ResnetBlock(model_channels*2, model_channels, embed_dim, activation_fn),
            ResnetBlock(model_channels*2, model_channels, embed_dim, activation_fn),
        ])

        self.out = torch.nn.Sequential(
            torch.nn.GroupNorm(32, model_channels),
            activation_fn(),
            torch.nn.Conv2d(model_channels, image_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        emb = self.embed(t)
        hs = []
        for module in self.input_blocks:
            if isinstance(module, TimestepBlock):
                x = module(x, emb)
            else:
                x = module(x)
            hs.append(x)
        x = self.middle_block(x, emb)
        for module in self.output_blocks:
            x = torch.cat([x, hs.pop()], 1)
            x = module(x, emb)
        return self.out(x)

def train(batch_size=128, epochs=80, lr=1e-3, model_channels=32, activation_fn=torch.nn.SiLU):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise_scheduler = NoiseScheduler().to(device)
    model = Model(model_channels=model_channels, activation_fn=activation_fn).to(device)
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
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

    torch.save(model.state_dict(), 'cifar-arch-model.pth')

def test(model_channels=32, activation_fn=torch.nn.SiLU):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise_scheduler = NoiseScheduler().to(device)
    model = Model(model_channels=model_channels, activation_fn=activation_fn).to(device)
    model.load_state_dict(torch.load('cifar-arch-model.pth', weights_only=True))
    model.eval()

    x = torch.randn(64, 3, 32, 32).to(device)
    for step in range(noise_scheduler.steps-1, -1, -1):
        with torch.no_grad():
            t = torch.tensor(step, device=device).expand(x.size(0),)
            pred_noise = model(x, t)
            x = noise_scheduler.sample_prev_step(x, t, pred_noise)

    x = denormalize(x).clamp(0, 1)

    # Create an image grid
    grid = make_grid(x, nrow=8, padding=2)
    grid = F.to_pil_image(grid)
    grid.save("cifar-arch-output.png")
    print("Image saved as cifar-arch-output.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Diffusion Process with Configurable Parameters")
    parser.add_argument('command', choices=['train', 'test', 'eval'], help="Command to execute")
    parser.add_argument('--batch-size', type=int, default=128, help="Batch size")
    parser.add_argument('--epochs', type=int, default=120, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--model-channels', type=int, default=32, help="Number of channels in the model")
    parser.add_argument('--activation', type=str, default='silu', choices=ACTIVATION_FUNCTIONS.keys(), help="Activation function")
    args = parser.parse_args()

    activation_fn = ACTIVATION_FUNCTIONS[args.activation]

    if args.command == 'train':
        train(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, model_channels=args.model_channels,
              activation_fn=activation_fn)
    elif args.command == 'test':
        test(model_channels=args.model_channels, activation_fn=activation_fn)
