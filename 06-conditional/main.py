import argparse
import math
import os
import shutil
import time
from abc import abstractmethod
from collections import namedtuple
from functools import partial

import torch
import torchvision
import torchvision.transforms.functional as F
from matplotlib import pyplot as plt
from torch.optim.swa_utils import get_ema_multi_avg_fn
from torchvision.utils import make_grid
from tqdm import trange

ACTIVATION_FUNCTIONS = {
    'relu': torch.nn.ReLU,
    'silu': torch.nn.SiLU,
    'leakyrelu': torch.nn.LeakyReLU,
    'gelu': torch.nn.GELU,
}

Dataset = namedtuple('Dataset', ('name', 'cls', 'image_channels', 'resolution', 'num_classes'))
dataset = [
    Dataset('mnist', partial(torchvision.datasets.MNIST, root='./data', train=True, download=True), 1, 28, 10),
    Dataset('cifar10', partial(torchvision.datasets.CIFAR10, root='./data', train=True, download=True), 3, 32, 10),
    Dataset('cifar100', partial(torchvision.datasets.CIFAR100, root='./data', train=True, download=True), 3, 32, 100),
    Dataset('celeba', partial(torchvision.datasets.CelebA, root='./data', split='train', download=True), 3, 64, 40),
]
DATASETS = {d.name: d for d in dataset}

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
    def __init__(self, in_channels, out_channels, embed_channels, activation_fn=torch.nn.SiLU, dropout=0.1):
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
            torch.nn.Dropout(dropout),
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

class AttentionBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.norm = torch.nn.GroupNorm(32, in_channels)
        self.qkv = torch.nn.Conv1d(in_channels, in_channels*3, kernel_size=1)
        self.out = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        _input = x.view(b, c, -1)
        x = self.norm(_input)
        qkv = self.qkv(x).permute(0, 2, 1)
        q, k, v = torch.chunk(qkv, 3, dim=2)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.permute(0, 2, 1)
        x = self.out(x)
        x = x + _input
        return x.view(b, c, h, w)


class Model(torch.nn.Module):
    def __init__(self,
                 image_channels=3,
                 model_channels=128,
                 activation_fn=torch.nn.SiLU,
                 num_res_blocks=2,
                 channel_mult=(1, 2, 2, 2),
                 dropout=0.1,
                 attention_resolutions=(2,),
                 num_classes=10,
                 conditional=True):
        super(Model, self).__init__()

        embed_dim = model_channels * 4
        self.embed = torch.nn.Sequential(
            PositionalEncoding(model_channels),
            torch.nn.Linear(model_channels, embed_dim),
            activation_fn(),
            torch.nn.Linear(embed_dim, embed_dim),
        )

        if conditional:
            self.class_emb = torch.nn.Embedding(num_classes, embed_dim)

        self.input_blocks = torch.nn.ModuleList([torch.nn.Conv2d(image_channels, model_channels, kernel_size=3, padding=1)])
        channels = [model_channels]
        ds = 1
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                in_ch = channels[-1]
                block = TimestepBlockSequential(ResnetBlock(in_ch, out_ch, embed_dim, activation_fn, dropout))
                if ds in attention_resolutions:
                    block.append(AttentionBlock(out_ch))
                self.input_blocks.append(block)
                channels.append(out_ch)
            if level < len(channel_mult) - 1:
                self.input_blocks.append(Downsample(out_ch, out_ch))
                channels.append(out_ch)
                ds *= 2

        self.middle_block = TimestepBlockSequential()
        out_ch = model_channels * channel_mult[-1]
        for _ in range(num_res_blocks):
            self.middle_block.append(ResnetBlock(out_ch, out_ch, embed_dim, activation_fn, dropout))

        self.output_blocks = torch.nn.ModuleList()
        for level, mult in enumerate(reversed(channel_mult)):
            for i in range(num_res_blocks + 1):
                in_ch = out_ch + channels.pop()
                out_ch = model_channels * mult
                if level == len(channel_mult) - 1 and i == num_res_blocks:
                    out_ch = model_channels
                block = TimestepBlockSequential(ResnetBlock(in_ch, out_ch, embed_dim, activation_fn, dropout))
                if i == num_res_blocks and level < len(channel_mult) - 1:
                    block.append(Upsample(out_ch, out_ch))
                self.output_blocks.append(block)

        self.out = torch.nn.Sequential(
            torch.nn.GroupNorm(32, model_channels),
            activation_fn(),
            torch.nn.Conv2d(model_channels, image_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t, class_idx):
        emb_t = self.embed(t)
        emb_class = self.class_emb(class_idx)
        emb = emb_t + emb_class
        hs = []
        for module in self.input_blocks:
            if isinstance(module, TimestepBlock):
                x = module(x, emb)
            else:
                x = module(x)
            hs.append(x)
        x = self.middle_block(x, emb)
        for module in self.output_blocks:
            h = hs.pop()
            x = torch.cat([x, h], 1)
            x = module(x, emb)
        return self.out(x)

def train(batch_size=128,
          epochs=80,
          lr=1e-3,
          warmup=0,
          grad_clip=None,
          ema_decay=0.9999,
          model_channels=32,
          activation_fn=torch.nn.SiLU,
          num_res_blocks=2,
          channel_mult=(1, 2, 2, 2),
          hflip=True,
          dropout=0.1,
          attention_resolutions=(2,),
          gpu=None,
          save_checkpoints=True,
          log_interval=10,
          output_dir='output',
          dataset='cifar10',
          conditional=True,
          resolution=None):
    device = torch.device(f'cuda:{gpu}' if gpu is not None else 'cpu')
    dataset = DATASETS[dataset]
    noise_scheduler = NoiseScheduler().to(device)
    model = Model(image_channels=dataset.image_channels,
                  model_channels=model_channels,
                  activation_fn=activation_fn,
                  num_res_blocks=num_res_blocks,
                  channel_mult=channel_mult,
                  dropout=dropout,
                  attention_resolutions=attention_resolutions,
                  num_classes=dataset.num_classes,
                  conditional=conditional)
    model = model.to(device)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))

    transforms = []
    resolution = resolution or dataset.resolution
    if resolution != dataset.resolution:
        transforms.append(torchvision.transforms.Resize(resolution))
    if hflip:
        transforms.append(torchvision.transforms.RandomHorizontalFlip())
    transforms.extend([
        torchvision.transforms.ToTensor(),
        normalize
    ])
    transform = torchvision.transforms.Compose(transforms)
    dataset = dataset.cls(root='./data', train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if warmup > 0:
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1e-9, 1, total_iters=warmup)
    criterion = torch.nn.MSELoss()

    # Train
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    if save_checkpoints:
        os.makedirs(checkpoint_dir, exist_ok=True)

    start = time.time()
    loss_history = []
    for epoch in range(1, epochs+1):
        model.train()
        loss_epoch = 0
        n = 0
        for x, cls in data_loader:
            x = x.to(device)
            cls = cls.to(device)
            optimizer.zero_grad()
            noise = torch.randn_like(x, device=device)
            t = torch.randint(0, noise_scheduler.steps, (x.size(0),), device=device)
            x_t = noise_scheduler.add_noise(x, t, noise)
            if conditional:
                pred_noise = model(x_t, t, cls)
            else:
                pred_noise = model(x_t, t)
            loss = criterion(pred_noise, noise)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            ema.update_parameters(model)
            loss_epoch += loss.item()
            n += x.size(0)
        loss_epoch /= n

        loss_history.append(loss_epoch)

        formatted_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
        print(f"Epoch {epoch:04d}, Loss {loss_epoch:.6f}, Time {formatted_time}")
        plot_loss(loss_history)

        model_path = os.path.join(output_dir, 'model.pth')
        ema_path = os.path.join(output_dir, 'model-ema.pth')
        torch.save(model.state_dict(), model_path)
        torch.save(ema.state_dict(), ema_path)

        if save_checkpoints:
            # copy the to checkpoint folder
            shutil.copy(model_path, f'checkpoints/model-{epoch:04d}.pth')
            shutil.copy(ema_path, f'checkpoints/model-ema-{epoch:04d}.pth')

        if epoch % log_interval == 0:
            file_path = os.path.join(output_dir, f'img/{epoch:04d}.png')
            _test(device, noise_scheduler, model, file_path)
            ema_file_path = os.path.join(output_dir, f'img-ema/{epoch:04d}.png')
            _test(device, noise_scheduler, ema, ema_file_path)

def plot_loss(loss_history, output_dir='output'):
    csv_path = os.path.join(output_dir, 'loss.csv')
    with open(csv_path, 'w') as f:
        f.write('epoch,loss\n')
        for i, loss in enumerate(loss_history):
            f.write(f'{i+1},{loss}\n')

    ema_loss = [loss_history[0]]
    for loss in loss_history[1:]:
        ema_loss.append(0.9 * ema_loss[-1] + 0.1 * loss)

    plt.plot(loss_history, color='blue', label='Loss')
    plt.plot(ema_loss, color='red', linestyle='dashed', label='EMA Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    plt.close()

def _test(device, noise_scheduler, model, file_path="img.png", progress=False, dataset='cifar10', conditional=True):
    # Use seed
    torch.manual_seed(0)
    dataset = DATASETS[dataset]
    n, nr = 256, 16
    classes = None
    if conditional and dataset.name in ['cifar10', 'mnist']:
        n, nr = 100, 10
        classes = torch.arange(10).repeat_interleave(10).to(device)
    elif conditional and dataset.name == 'cifar100':
        n, nr = 100, 10
        classes = torch.arange(100).to(device)
    elif conditional:
        raise ValueError(f"Conditional model is not supported for {dataset.name}")

    x = torch.randn(n, dataset.image_channels, dataset.resolution, dataset.resolution, device=device)

    if progress:
        steps = trange(noise_scheduler.steps-1, -1, -1)
    else:
        steps = range(noise_scheduler.steps-1, -1, -1)

    for step in steps:
        with torch.no_grad():
            t = torch.tensor(step, device=device).expand(x.size(0),)
            if conditional:
                pred_noise = model(x, t, classes)
            else:
                pred_noise = model(x, t)
            x = noise_scheduler.sample_prev_step(x, t, pred_noise)

    x = denormalize(x).clamp(0, 1)

    # Create an image grid
    grid = make_grid(x, nrow=nr, padding=2)
    grid = F.to_pil_image(grid)
    grid.save(file_path)
    torch.seed()  # Reset seed

def test(model_channels=32,
         activation_fn=torch.nn.SiLU,
         num_res_blocks=2,
         channel_mult=(1, 2, 2, 2),
         dropout=0.1,
         attention_resolutions=(2,),
         gpu=None,
         model_path='model.pth',
         file_path='img.png',
         dataset='cifar10',
         conditional=True):
    dataset = DATASETS[dataset]
    device = torch.device(f'cuda:{gpu}' if gpu is not None else 'cpu')
    noise_scheduler = NoiseScheduler().to(device)
    model = Model(image_channels=dataset.image_channels,
                  model_channels=model_channels,
                  activation_fn=activation_fn,
                  num_res_blocks=num_res_blocks,
                  channel_mult=channel_mult,
                  dropout=dropout,
                  attention_resolutions=attention_resolutions,
                  num_classes=dataset.num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()
    _test(device, noise_scheduler, model, file_path, progress=True, dataset=dataset.name, conditional=conditional)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Diffusion Process with Configurable Parameters")
    parser.add_argument('command', choices=['train', 'test', 'eval'], help="Command to execute")
    parser.add_argument('--dataset', choices=['mnist', 'cifar10', 'cifar100', 'celeba'], default='cifar10', help="Dataset to use")
    parser.add_argument('--batch-size', type=int, default=128, help="Batch size")
    parser.add_argument('--epochs', type=int, default=2000, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate")
    parser.add_argument('--image-channels', type=int, default=3, help="Number of image channels")
    parser.add_argument('--model-channels', type=int, default=128, help="Number of channels in the model")
    parser.add_argument('--activation', type=str, default='silu', choices=ACTIVATION_FUNCTIONS.keys(), help="Activation function")
    parser.add_argument('--num-res-blocks', type=int, default=2, help="Number of residual blocks")
    parser.add_argument('--channel-mult', type=int, nargs=4, default=(1, 2, 2, 2), help="Channel multipliers")
    parser.add_argument('--hflip', action='store_true', help="Use horizontal flips")
    parser.add_argument('--no-hflip', dest='hflip', action='store_false', help="Do not use horizontal flips")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate")
    parser.add_argument('--attention-resolutions', type=int, nargs='+', default=(2,), help="Resolutions to apply attention")
    parser.add_argument('--gpu', type=int, default=None, help="GPU index")
    parser.add_argument('--model', type=str, default='model.pth', help="Model file")
    parser.add_argument('--save-checkpoints', action='store_true', help="Save model checkpoints")
    parser.add_argument('--log-interval', type=int, default=10, help="Image log interval")
    parser.add_argument('--output-dir', type=str, default='output', help="Output directory")
    parser.add_argument('--file-path', type=str, default='img.png', help="Output file path")
    parser.add_argument('--conditional', action='store_true', help="Use conditional model")
    args = parser.parse_args()

    activation_fn = ACTIVATION_FUNCTIONS[args.activation]

    if args.command == 'train':
        train(batch_size=args.batch_size,
              epochs=args.epochs,
              lr=args.lr,
              model_channels=args.model_channels,
              activation_fn=activation_fn,
              num_res_blocks=args.num_res_blocks,
              channel_mult=args.channel_mult,
              hflip=args.hflip,
              dropout=args.dropout,
              attention_resolutions=args.attention_resolutions,
              gpu=args.gpu,
              save_checkpoints=args.save_checkpoints,
              log_interval=args.log_interval,
              output_dir=args.output_dir,
              dataset=args.dataset,
              conditional=args.conditional)
    elif args.command == 'test':
        test(model_channels=args.model_channels,
             activation_fn=activation_fn,
             num_res_blocks=args.num_res_blocks,
             channel_mult=args.channel_mult,
             dropout=args.dropout,
             attention_resolutions=args.attention_resolutions,
             gpu=args.gpu,
             model_path=args.model,
             file_path=args.file_path,
             dataset=args.dataset,
             conditional=args.conditional)