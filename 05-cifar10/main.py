import argparse
import math
import time
from abc import abstractmethod

import torch
import torchvision
import torchvision.transforms.functional as F
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from tqdm import trange


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

class TimestepBlock(torch.nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        raise NotImplementedError

class ResnetBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, embed_dim, dropout=0.1):
        super(ResnetBlock, self).__init__()
        self.norm1 = torch.nn.GroupNorm(32, in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.proj = torch.nn.Linear(embed_dim, out_channels)
        self.norm2 = torch.nn.GroupNorm(32, out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = torch.nn.Identity()

    def forward(self, x, emb):
        _input = x
        x = self.norm1(x)
        x = torch.nn.functional.silu(x)
        x = self.conv1(x)
        emb_proj = self.proj(emb).view(-1, x.size(1), 1, 1)
        x = x + emb_proj
        x = self.norm2(x)
        x = torch.nn.functional.silu(x)
        x = self.dropout(x)
        x = self.conv2(x)
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

class AttentionBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.norm = torch.nn.GroupNorm(32, in_channels)
        self.qkv = torch.nn.Conv1d(in_channels, in_channels*3, kernel_size=1)
        self.attn = torch.nn.MultiheadAttention(in_channels, 1, batch_first=True)
        self.out = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        # print(f'b: {b}, c: {c}, h: {h}, w: {w}')
        _input = x.view(b, c, -1)
        x = self.norm(_input)
        # print(f'x.shape: {x.shape}')
        qkv = self.qkv(x).permute(0, 2, 1)
        q, k, v = torch.chunk(qkv, 3, dim=2)
        # print(f'q.shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}')
        x, _ = self.attn(q, k, v, need_weights=False)
        x = x.permute(0, 2, 1)
        x = self.out(x)
        # print(f'x.shape: {x.shape}')
        x = x + _input
        return x.view(b, c, h, w)


class Model(torch.nn.Module):
    def __init__(self, num_steps=1000, ch=128, dropout=0.1):
        super(Model, self).__init__()

        embed_dim = ch * 4
        self.embed = torch.nn.Sequential(
            PositionalEncoding(ch, num_steps),
            torch.nn.Linear(ch, embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.SiLU(),
        )

        self.input_blocks = torch.nn.ModuleList([
            torch.nn.Conv2d(3, ch, kernel_size=3, padding=1),

            ResnetBlock(ch, ch, embed_dim, dropout),
            ResnetBlock(ch, ch, embed_dim, dropout),
            Downsample(ch, ch),

            TimestepBlockSequential(
                ResnetBlock(ch, ch*2, embed_dim, dropout),
                AttentionBlock(ch*2)
            ),
            TimestepBlockSequential(
                ResnetBlock(ch*2, ch*2, embed_dim, dropout),
                AttentionBlock(ch*2)
            ),
            Downsample(ch*2, ch*2),

            ResnetBlock(ch*2, ch*2, embed_dim, dropout),
            ResnetBlock(ch*2, ch*2, embed_dim, dropout),
            Downsample(ch*2, ch*2),

            ResnetBlock(ch*2, ch*2, embed_dim, dropout),
            ResnetBlock(ch*2, ch*2, embed_dim, dropout),
        ])

        self.middle_block = TimestepBlockSequential(
            ResnetBlock(ch*2, ch*2, embed_dim, dropout),
            AttentionBlock(ch*2),
            ResnetBlock(ch*2, ch*2, embed_dim, dropout),
        )

        self.output_blocks = torch.nn.ModuleList([
            ResnetBlock(ch*4, ch*2, embed_dim, dropout),
            ResnetBlock(ch*4, ch*2, embed_dim, dropout),
            TimestepBlockSequential(
                ResnetBlock(ch*4, ch*2, embed_dim, dropout),
                Upsample(ch*2, ch*2),
            ),

            ResnetBlock(ch*4, ch*2, embed_dim, dropout),
            ResnetBlock(ch*4, ch*2, embed_dim, dropout),
            TimestepBlockSequential(
                ResnetBlock(ch*4, ch*2, embed_dim, dropout),
                Upsample(ch*2, ch*2),
            ),

            TimestepBlockSequential(
                ResnetBlock(ch*4, ch*2, embed_dim, dropout),
                AttentionBlock(ch*2)
            ),
            TimestepBlockSequential(
                ResnetBlock(ch*4, ch*2, embed_dim, dropout),
                AttentionBlock(ch*2)
            ),
            TimestepBlockSequential(
                ResnetBlock(ch*3, ch*2, embed_dim, dropout),
                AttentionBlock(ch*2),
                Upsample(ch*2, ch*2),
            ),

            ResnetBlock(ch*3, ch, embed_dim, dropout),
            ResnetBlock(ch*2, ch, embed_dim, dropout),
            ResnetBlock(ch*2, ch, embed_dim, dropout),
        ])

        self.out = torch.nn.Sequential(
            torch.nn.GroupNorm(32, ch),
            torch.nn.SiLU(),
            torch.nn.Conv2d(ch, 3, kernel_size=3, padding=1)
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

class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] -= (1 - self.decay) * (self.shadow[name] - param.data)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

def train(batch_size=128, epochs=80, lr=2e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise_scheduler = NoiseScheduler(steps=1000, beta_start=1e-4, beta_end=0.02).to(device)
    model = Model().to(device)
    ema = EMA(model)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        normalize
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # Train
    start = time.time()
    loss_history = []
    _test(device, noise_scheduler, model, epoch=0)
    for epoch in range(1, epochs+1):
        model.train()
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
            ema.update()
            loss_epoch += loss.item()
            n += x.size(0)
        loss_epoch /= n

        loss_history.append(loss_epoch)

        formatted_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
        print(f"Epoch {epoch:04d}, Loss {loss_epoch:.6f}, Time {formatted_time}")
        plot_loss(loss_history)

        if epoch % 20 == 0:
            torch.save(model.state_dict(), 'part-i-cifar-full-model.pth')
            ema.apply_shadow()
            torch.save(model.state_dict(), 'part-i-cifar-full-model-ema.pth')
            _test(device, noise_scheduler, model, epoch)
            model.train()
            ema.restore()

    print(f"Epoch {epoch}, Loss {loss_epoch}, Time {time.time() - start}")
    torch.save(model.state_dict(), 'part-i-cifar-full-model.pth')
    ema.apply_shadow()
    torch.save(model.state_dict(), 'part-i-cifar-full-model-ema.pth')
    _test(device, noise_scheduler, model, epoch)

def plot_loss(loss_history):
    ema_loss = [loss_history[0]]
    for loss in loss_history[1:]:
        ema_loss.append(0.9 * ema_loss[-1] + 0.1 * loss)

    plt.plot(loss_history, color='blue', label='Loss')
    plt.plot(ema_loss, color='red', linestyle='dashed', label='EMA Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid()
    plt.savefig('part-i-cifar-full-loss.png')
    plt.close()
    # plt.show()

def _test(device, noise_scheduler, model, epoch=None, progress=False):
    # Use seed
    torch.manual_seed(0)

    x = torch.randn(256, 3, 32, 32).to(device)

    if progress:
        steps = trange(noise_scheduler.steps-1, -1, -1)
    else:
        steps = range(noise_scheduler.steps-1, -1, -1)

    for step in steps:
        with torch.no_grad():
            t = torch.tensor(step, device=device).expand(x.size(0),)
            pred_noise = model(x, t)
            x = noise_scheduler.sample_prev_step(x, t, pred_noise)

    x = denormalize(x).clamp(0, 1)

    # Create an image grid
    grid = make_grid(x, nrow=16, padding=2)
    grid = F.to_pil_image(grid)
    suffix = f"-{epoch:04d}" if epoch is not None else ""
    filename = f"part-i-cifar-full-output{suffix}.png"
    grid.save(filename)
    # print(f"Image saved as {filename}")
    torch.seed() # Reset seed

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise_scheduler = NoiseScheduler(steps=1000, beta_start=1e-4, beta_end=0.02).to(device)
    model = Model().to(device)
    model.load_state_dict(torch.load('part-i-cifar-full-model.pth', weights_only=True))
    model.eval()
    _test(device, noise_scheduler, model, progress=True)

def eval(batch_size):
    from eval import eval_nll

    # Load model and evaluate NLL
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise_scheduler = NoiseScheduler().to(device)
    model = Model().to(device)
    model.load_state_dict(torch.load('part-i-cifar-attn-model.pth', weights_only=True))
    model.eval()

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    eval_nll(model, test_loader, noise_scheduler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Diffusion Process with Configurable Parameters")
    parser.add_argument('command', choices=['train', 'test', 'eval'], help="Command to execute")
    parser.add_argument('--batch-size', type=int, default=128, help="Batch size")
    parser.add_argument('--epochs', type=int, default=2000, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate")
    parser.add_argument('--fid-samples', type=int, default=50000, help="Number of samples for FID calculation")
    args = parser.parse_args()

    if args.command == 'train':
        train(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)
    elif args.command == 'test':
        test()
    elif args.command == 'eval':
        eval(batch_size=args.batch_size)
