import argparse
import math
import os
import time
import warnings
from abc import abstractmethod

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torchinfo
import torchvision
import torchvision.transforms.functional as F
from torchvision.utils import make_grid

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

def train(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
    if args.multiprocessing_distributed:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    model = Model()

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    noise_scheduler = NoiseScheduler(steps=1000, beta_start=1e-4, beta_end=0.02).to(device)
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        normalize
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=sampler)

    # Train
    start = time.time()
    for epoch in range(args.epochs):
        if args.distributed:
            sampler.set_epoch(epoch)

        loss_epoch = 0
        n = 0
        for x, _ in data_loader:
            x = x.to(device, non_blocking=True)
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

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            print(f"Epoch {epoch}, Loss {loss_epoch}")

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss {loss_epoch}, Time {time.time() - start}")
                torch.save(model.state_dict(), 'part-j-cifar-full-distributed-model.pth')
                _test(device, noise_scheduler, model)
                model.train()

    print(f"Epoch {epoch}, Loss {loss_epoch}, Time {time.time() - start}")
    _test(device, noise_scheduler, model)
    torch.save(model.state_dict(), 'part-j-cifar-full-distributed-model.pth')

def _test(device, noise_scheduler, model):
    x = torch.randn(256, 3, 32, 32).to(device)
    for step in range(noise_scheduler.steps-1, -1, -1):
        with torch.no_grad():
            t = torch.tensor(step, device=device).expand(x.size(0),)
            pred_noise = model(x, t)
            x = noise_scheduler.sample_prev_step(x, t, pred_noise)

    x = denormalize(x).clamp(0, 1)

    # Create an image grid
    grid = make_grid(x, nrow=16, padding=2)
    grid = F.to_pil_image(grid)
    grid.save("part-i-cifar-full-output.png")
    print("Image saved as part-j-cifar-full-distributed-output.png")

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise_scheduler = NoiseScheduler(steps=1000, beta_start=1e-4, beta_end=0.02).to(device)
    model = Model().to(device)
    model.load_state_dict(torch.load('part-j-cifar-full-distributed-odel.pth', weights_only=True))
    model.eval()
    _test(device, noise_scheduler, model)

def eval(batch_size):
    from eval import eval_nll

    # Load model and evaluate NLL
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise_scheduler = NoiseScheduler().to(device)
    model = Model().to(device)
    model.load_state_dict(torch.load('part-j-cifar-full-distributed-model.pth', weights_only=True))
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
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    args = parser.parse_args()

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
        if ngpus_per_node == 1 and args.dist_backend == "nccl":
            warnings.warn("nccl backend >=2.5 requires GPU count>1, see https://github.com/NVIDIA/nccl/issues/103 perhaps use 'gloo'")
    else:
        ngpus_per_node = 1

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(train, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        train(args.gpu, ngpus_per_node, args)
