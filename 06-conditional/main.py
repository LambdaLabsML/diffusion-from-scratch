import argparse
import csv
import math
import os
import shutil
import time
from abc import abstractmethod
from collections import namedtuple
from functools import partial
from typing import Tuple, Any, Optional, Union, Callable, List

import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from torch.optim.swa_utils import get_ema_multi_avg_fn
from torch.utils.data import RandomSampler, Sampler
from torchvision.utils import make_grid
from tqdm import trange, tqdm

CONFIGS = {
    "cifar10": [
        'train',
        '--dataset', 'cifar10',
        '--batch-size', '128',
        '--grad-clip', '1',
        '--lr', '2e-4',
        '--warmup', '5000',
        '--steps', '800_000',
        '--val-interval', '2000',
        '--model-channels', '128',
        '--channel-mult', '1', '2', '2', '2',
        '--num-res-blocks', '2',
        '--attention-resolutions', '2',
        '--dropout', '0.1',
        '--hflip',
        '--save-checkpoints',
        '--log-interval', '5',
        '--progress',
    ],
    "cifar10-cond": [
        'train',
        '--dataset', 'cifar10',
        '--conditional',
        '--batch-size', '128',
        '--grad-clip', '1',
        '--lr', '2e-4',
        '--warmup', '5000',
        '--steps', '800_000',
        '--val-interval', '2000',
        '--model-channels', '128',
        '--channel-mult', '1', '2', '2', '2',
        '--num-res-blocks', '2',
        '--attention-resolutions', '2',
        '--dropout', '0.1',
        '--hflip',
        '--save-checkpoints',
        '--log-interval', '5',
        '--progress',
    ],
    'celeba-64': [
        'train',
        '--dataset', 'celeba',
        '--resolution', '64',
        '--batch-size', '64',
        '--grad-clip', '1',
        '--lr', '2e-5',
        '--warmup', '5000',
        '--steps', '500_000',
        '--val-interval', '1000',
        '--model-channels', '128',
        '--channel-mult', '1', '1', '2', '2', '4', '4',
        '--num-res-blocks', '2',
        '--attention-resolutions', '16',
        '--dropout', '0.0',
        '--hflip',
        # '--save-checkpoints',
        '--log-interval', '5',
        '--progress',
    ],
    'celeba-128': [
        'train',
        '--dataset', 'celeba',
        '--resolution', '128',
        '--batch-size', '64',
        '--grad-clip', '1',
        '--lr', '2e-5',
        '--warmup', '5000',
        '--steps', '500_000',
        '--val-interval', '1000',
        '--model-channels', '128',
        '--channel-mult', '1', '1', '2', '2', '4', '4',
        '--num-res-blocks', '2',
        '--attention-resolutions', '16',
        '--dropout', '0.0',
        '--hflip',
        # '--save-checkpoints',
        '--log-interval', '5',
        '--progress',
    ],
    'celeba-256': [
        'train',
        '--dataset', 'celeba',
        '--resolution', '256',
        '--batch-size', '16',
        '--grad-clip', '1',
        '--grad-accum', '4',
        '--lr', '2e-5',
        '--warmup', '20_000',
        '--steps', '2_000_000',
        '--val-interval', '4000',
        '--model-channels', '128',
        '--channel-mult', '1', '1', '2', '2', '4', '4',
        '--num-res-blocks', '2',
        '--attention-resolutions', '16',
        '--dropout', '0.0',
        '--hflip',
        # '--save-checkpoints',
        '--log-interval', '5',
        '--progress',
    ],
    'mnist': [
        'train',
        '--dataset', 'mnist',
        '--batch-size', '256',
        '--grad-clip', '1',
        '--lr', '1e-3',
        '--warmup', '5000',
        '--steps', '200_000',
        '--val-interval', '4000',
        '--model-channels', '32',
        '--channel-mult', '1', '1', '2',
        '--num-res-blocks', '1',
        '--attention-resolutions', '2',
        '--dropout', '0.1',
        '--save-checkpoints',
        '--log-interval', '1',
        '--progress',
    ],
    'mnist-cond': [
        'train',
        '--dataset', 'mnist',
        '--conditional',
        '--batch-size', '256',
        '--grad-clip', '1',
        '--lr', '1e-3',
        '--warmup', '5000',
        '--steps', '200_000',
        '--val-interval', '4000',
        '--model-channels', '32',
        '--channel-mult', '1', '1', '2',
        '--num-res-blocks', '1',
        '--attention-resolutions', '2',
        '--dropout', '0.1',
        '--save-checkpoints',
        '--log-interval', '1',
        '--progress',
    ]
}

CSV = namedtuple("CSV", ["header", "index", "data"])


class CelebAHQ(torchvision.datasets.VisionDataset):
    def __init__(self, root,
                 target_type: Union[List[str], str] = "attr",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        attr = self._load_csv("CelebAMask-HQ-attribute-anno.txt", header=1)
        self.filename = attr.index
        self.attr = attr.data
        # map from {-1, 1} to {0, 1}
        self.attr = torch.div(self.attr + 1, 2, rounding_mode="floor")
        self.attr_names = attr.header

    def _load_csv(
            self,
            filename: str,
            header: Optional[int] = None,
    ) -> CSV:
        with open(os.path.join(self.root, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1:]
        else:
            headers = []

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = Image.open(os.path.join(self.root, "CelebA-HQ-img-orig", self.filename[index]))

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            # elif t == "identity":
            #     target.append(self.identity[index, 0])
            # elif t == "bbox":
            #     target.append(self.bbox[index, :])
            # elif t == "landmarks":
            #     target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError(f'Target type "{t}" is not recognized.')

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)


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
    Dataset('celeba', partial(CelebAHQ, root='./data/CelebAMask-HQ'), 3, 1024, 40),
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

    def sample_prev_step(self, xt, t, pred_noise, z=None):
        if z is None:
            z = torch.randn_like(xt)
        t = t.view(-1, 1, 1, 1)
        z[t.expand_as(z) == 0] = 0

        mean = (1 / torch.sqrt(self.alpha[t])) * (xt - (self.beta[t] / torch.sqrt(1 - self.alpha_bar[t])) * pred_noise)
        var = ((1 - self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t])) * self.beta[t]
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
        self.qkv = torch.nn.Conv1d(in_channels, in_channels * 3, kernel_size=1)
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
        self.conditional = conditional

        embed_dim = model_channels * 4
        self.embed = torch.nn.Sequential(
            PositionalEncoding(model_channels),
            torch.nn.Linear(model_channels, embed_dim),
            activation_fn(),
            torch.nn.Linear(embed_dim, embed_dim),
        )

        if self.conditional:
            self.class_emb = torch.nn.Embedding(num_classes, embed_dim)

        self.input_blocks = torch.nn.ModuleList(
            [torch.nn.Conv2d(image_channels, model_channels, kernel_size=3, padding=1)])
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

    def forward(self, x, t, class_idx=None, emb_class=None):
        emb = self.embed(t)

        if self.conditional:
            if emb_class is None:
                emb_class = self.class_emb(class_idx)
            emb = emb + emb_class
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

class ContinuousSampler(Sampler):
    def __init__(self, data_source, batch_size, val_interval, shuffle=True):
        super(ContinuousSampler, self).__init__()
        self.shuffle = shuffle
        self.iteration = -1
        self.items_per_epoch = batch_size * val_interval
        if shuffle:
            self.indexes = torch.randperm(len(data_source))
        else:
            self.indexes = torch.arange(len(data_source))

    def __len__(self):
        return self.items_per_epoch

    def __iter__(self):
        for i in range(self.items_per_epoch):
            self.iteration += 1
            if self.iteration == len(self.indexes):
                if self.shuffle:
                    self.indexes = torch.randperm(len(self.indexes))
                self.iteration = 0
            yield self.indexes[self.iteration % len(self.indexes)]

def train(batch_size=128,
          epochs=80,
          steps=None,
          val_interval=None,
          lr=1e-3,
          warmup=0,
          grad_clip=None,
          grad_accum=1,
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
          resolution=None,
          progress=False):
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
    ds = dataset.cls(transform=transform)

    sampler = None
    # print(f'val_interval: {val_interval}, steps: {steps}')
    if val_interval is not None:
        sampler = ContinuousSampler(ds, batch_size, val_interval, shuffle=True)
    data_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    # print(f'Length of dataset: {len(ds)}, length of dataloader: {len(data_loader)}')
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
    if log_interval != 0:
        os.makedirs(os.path.join(output_dir, 'img'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'img-ema'), exist_ok=True)

    start = time.time()
    loss_history = []

    if steps is not None:
        epochs = math.ceil(steps / len(data_loader))
        print(f'Training for {steps} steps ({epochs} epochs)')
    else:
        calc_steps = len(data_loader) * epochs
        print(f'Training for {epochs} epochs ({calc_steps} steps)')
    iteration = 1
    for epoch in range(1, epochs + 1):
        model.train()
        loss_epoch = 0
        n = 0
        if progress:
            batches = pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch:04d}')
        else:
            batches = enumerate(data_loader)

        for batch_idx, (x, cls) in batches:
            if batch_idx % grad_accum == 0:
                optimizer.zero_grad()
            x = x.to(device)
            cls = cls.to(device)
            noise = torch.randn_like(x, device=device)
            t = torch.randint(0, noise_scheduler.steps, (x.size(0),), device=device)
            x_t = noise_scheduler.add_noise(x, t, noise)
            if conditional:
                pred_noise = model(x_t, t, cls)
            else:
                pred_noise = model(x_t, t)
            loss = criterion(pred_noise, noise) / grad_accum
            loss.backward()
            if (batch_idx + 1) % grad_accum == 0 or batch_idx == len(data_loader) - 1:
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                ema.update_parameters(model)
            n += x.size(0)
            loss_epoch += loss.item()
            loss_history.append(loss.item())
            if progress:
                psnr = 20 * math.log10(noise.abs().max().item()) - 10 * math.log10(loss.item())
                avg_loss_epoch = loss_epoch / (batch_idx + 1)
                pbar.set_postfix(
                    {'It': iteration, 'AvgLoss': f'{avg_loss_epoch:.4f}', 'Loss': f'{loss.item():.4f}', 'PSNR': f'{psnr:.2f}'})
            iteration += 1
        loss_epoch /= len(data_loader)

        formatted_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
        if progress:
            pbar.desc = f'Epoch={epoch:04d}, L={loss_epoch:.4f}, {formatted_time}'
            pbar.close()
        else:
            print(f"Epoch {epoch:04d}, Loss {loss_epoch:.6f}, Time {formatted_time}")
        plot_loss(loss_history, output_dir, warmup)

        model_path = os.path.join(output_dir, 'model.pth')
        ema_path = os.path.join(output_dir, 'model-ema.pth')
        torch.save(model.state_dict(), model_path)
        torch.save(ema.state_dict(), ema_path)

        if save_checkpoints:
            # copy to the checkpoint folder
            shutil.copy(model_path, os.path.join(checkpoint_dir, f'model-{epoch:04d}.pth'))
            shutil.copy(ema_path, os.path.join(checkpoint_dir, f'model-ema-{epoch:04d}.pth'))

        if log_interval != 0 and epoch % log_interval == 0:
            file_path = os.path.join(output_dir, f'img/{epoch:04d}.png')
            _test(device, noise_scheduler, model, file_path, dataset=dataset.name, resolution=resolution,
                  conditional=conditional, progress=progress)
            ema_file_path = os.path.join(output_dir, f'img-ema/{epoch:04d}.png')
            _test(device, noise_scheduler, ema, ema_file_path, dataset=dataset.name, resolution=resolution,
                  conditional=conditional, progress=progress)

        if steps is not None and iteration > steps:
            break


def plot_loss(loss_history, output_dir='output', warmup=0):
    csv_path = os.path.join(output_dir, 'loss.csv')
    with open(csv_path, 'w') as f:
        f.write('iter,loss\n')
        for i, loss in enumerate(loss_history):
            f.write(f'{i + 1},{loss}\n')

    ema_loss = [loss_history[0]]
    for loss in loss_history[1:]:
        ema_loss.append(0.9 * ema_loss[-1] + 0.1 * loss)

    y_limit = None
    if len(loss_history) > warmup:
        y_limit = max(loss_history[warmup:]) * 1.05

    plt.plot(loss_history, color='blue', label='Loss')
    # plt.plot(ema_loss, color='red', linestyle='dashed', label='EMA Loss')
    plt.ylim(0, y_limit)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    plt.close()


def _test(device, noise_scheduler, model, file_path="img.png", progress=False, dataset='cifar10', resolution=None,
          conditional=True):
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
    elif resolution == 256:
        n, nr = 64, 8

    x = torch.randn(n, dataset.image_channels, resolution, resolution, device=device)

    if progress:
        steps = trange(noise_scheduler.steps - 1, -1, -1)
    else:
        steps = range(noise_scheduler.steps - 1, -1, -1)

    for step in steps:
        with torch.no_grad():
            t = torch.tensor(step, device=device).expand(x.size(0), )
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
         conditional=True,
         resolution=None):
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
    model = model.to(device)
    model.eval()
    _test(device, noise_scheduler, model, file_path, progress=True, dataset=dataset.name, resolution=resolution, conditional=conditional)


def load_config(parser, args):
    default_args = parser.parse_args([args.command])
    # print(f'Default Args: {default_args}')
    config = CONFIGS[args.config]
    config_args = parser.parse_args(config)
    # print(f'Config Args: {config_args}')
    default_items = set((k, str(v)) for k, v in vars(default_args).items())
    args_items = set((k, str(v)) for k, v in vars(args).items())
    config_items = set((k, str(v)) for k, v in vars(config_args).items())
    overrides = (args_items - default_items) - config_items

    for k, _ in overrides:
        v = getattr(args, k)
        # print(f'Overriding {k} with {v}')
        setattr(config_args, k, v)
    return config_args

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Simple Diffusion Process with Configurable Parameters")
    parser.add_argument('command', choices=['train', 'test', 'eval'], help="Command to execute")
    parser.add_argument('--dataset', choices=['mnist', 'cifar10', 'cifar100', 'celeba'], default='cifar10',
                        help="Dataset to use")
    parser.add_argument('--batch-size', type=int, default=128, help="Batch size")
    parser.add_argument('--epochs', type=int, default=2000, help="Number of epochs")
    parser.add_argument('--steps', type=int, default=None, help="Number of steps")
    parser.add_argument('--val-interval', type=int, default=None, help="Validation interval")
    parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate")
    parser.add_argument('--grad-clip', type=float, default=None, help="Gradient clipping")
    parser.add_argument('--grad-accum', type=int, default=1, help="Gradient accumulation")
    parser.add_argument('--warmup', type=int, default=0, help="Warmup steps")
    parser.add_argument('--ema-decay', type=float, default=0.9999, help="Exponential moving average decay")
    parser.add_argument('--model-channels', type=int, default=128, help="Number of channels in the model")
    parser.add_argument('--activation', type=str, default='silu', choices=ACTIVATION_FUNCTIONS.keys(),
                        help="Activation function")
    parser.add_argument('--num-res-blocks', type=int, default=2, help="Number of residual blocks")
    parser.add_argument('--channel-mult', type=int, nargs='+', default=(1, 2, 2, 2), help="Channel multipliers")
    parser.add_argument('--hflip', action='store_true', help="Use horizontal flips")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate")
    parser.add_argument('--attention-resolutions', type=int, nargs='+', default=(2,),
                        help="Resolutions to apply attention")
    parser.add_argument('--gpu', type=int, default=None, help="GPU index")
    parser.add_argument('--model', type=str, default='model.pth', help="Model file")
    parser.add_argument('--save-checkpoints', action='store_true', help="Save model checkpoints")
    parser.add_argument('--log-interval', type=int, default=10, help="Image log interval")
    parser.add_argument('--output-dir', type=str, default='output', help="Output directory")
    parser.add_argument('--file-path', type=str, default='img.png', help="Output file path")
    parser.add_argument('--conditional', action='store_true', help="Use conditional model")
    parser.add_argument('--resolution', type=int, default=None, help="Resolution to use")
    parser.add_argument('--progress', action='store_true', help="Show progress bar")
    parser.add_argument('--config', choices=CONFIGS.keys(), default=None, help="Configuration to use")

    args = parser.parse_args(args=args)
    if args.config is not None:
        args = load_config(parser, args)
    return args

def main(args=None):
    args = parse_args(args)

    activation_fn = ACTIVATION_FUNCTIONS[args.activation]

    if args.command == 'train':
        train(batch_size=args.batch_size,
              epochs=args.epochs,
              steps=args.steps,
              val_interval=args.val_interval,
              lr=args.lr,
              grad_clip=args.grad_clip,
              grad_accum=args.grad_accum,
              warmup=args.warmup,
              ema_decay=args.ema_decay,
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
              conditional=args.conditional,
              resolution=args.resolution,
              progress=args.progress)
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
             conditional=args.conditional,
             resolution=args.resolution)


if __name__ == "__main__":
    main()
