import argparse
import math
import os
import shutil
import time
from abc import abstractmethod

import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from tqdm import trange

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
                 num_classes=10):
        super(Model, self).__init__()

        embed_dim = model_channels * 4
        self.embed = torch.nn.Sequential(
            PositionalEncoding(model_channels),
            torch.nn.Linear(model_channels, embed_dim),
            activation_fn(),
            torch.nn.Linear(embed_dim, embed_dim),
        )

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

def train(batch_size=128,
          epochs=80,
          lr=1e-3,
          image_channels=3,
          model_channels=32,
          activation_fn=torch.nn.SiLU,
          num_res_blocks=2,
          channel_mult=(1, 2, 2, 2),
          hflip=True,
          dropout=0.1,
          attention_resolutions=(2,),
          gpu=None,
          save_checkpoints=True):
    device = torch.device(f'cuda:{gpu}' if gpu is not None and torch.cuda.is_available() else 'cpu')
    noise_scheduler = NoiseScheduler().to(device)
    model = Model(image_channels=image_channels,
                  model_channels=model_channels,
                  activation_fn=activation_fn,
                  num_res_blocks=num_res_blocks,
                  channel_mult=channel_mult,
                  dropout=dropout,
                  attention_resolutions=attention_resolutions)
    model = model.to(device)
    ema = EMA(model)

    transforms = []
    if hflip:
        transforms.append(torchvision.transforms.RandomHorizontalFlip())
    transforms.extend([
        torchvision.transforms.ToTensor(),
        normalize
    ])
    transform = torchvision.transforms.Compose(transforms)
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # Train
    if save_checkpoints:
        os.makedirs('checkpoints', exist_ok=True)

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
            pred_noise = model(x_t, t, cls)
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

        torch.save(model.state_dict(), 'model.pth')
        _test(device, noise_scheduler, model, epoch, ema=False)
        ema.apply_shadow()
        torch.save(model.state_dict(), 'model-ema.pth')
        _test(device, noise_scheduler, model, epoch, ema=True)
        ema.restore()

        if save_checkpoints:
            # copy the to checkpoint folder
            shutil.copy('model.pth', f'checkpoints/model-{epoch:04d}.pth')
            shutil.copy('model-ema.pth', f'checkpoints/model-ema-{epoch:04d}.pth')

    print(f"Epoch {epoch}, Loss {loss_epoch}, Time {time.time() - start}")
    torch.save(model.state_dict(), 'part-i-cifar-full-model.pth')
    ema.apply_shadow()
    torch.save(model.state_dict(), 'part-i-cifar-full-model-ema.pth')
    _test(device, noise_scheduler, model, epoch)

def plot_loss(loss_history):
    with open('cifar-output/output/part-a-cifar-conditional-loss.csv', 'w') as f:
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
    plt.legend()
    plt.grid()
    plt.savefig('loss.png')
    plt.close()
    # plt.show()

def _test(device, noise_scheduler, model, epoch=None, progress=False, ema=False):
    # Use seed
    torch.manual_seed(0)

    x = torch.randn(100, 1, 28, 28).to(device)
    # grid of classes, 10x 0-9 e.g.: [0, 0, ..., 0, 1, 1, ..., 1, ..., 9, 9, ..., 9]
    classes = torch.arange(10).repeat_interleave(10).to(device)

    if progress:
        steps = trange(noise_scheduler.steps-1, -1, -1)
    else:
        steps = range(noise_scheduler.steps-1, -1, -1)

    for step in steps:
        with torch.no_grad():
            t = torch.tensor(step, device=device).expand(x.size(0),)
            pred_noise = model(x, t, classes)
            x = noise_scheduler.sample_prev_step(x, t, pred_noise)

    x = denormalize(x).clamp(0, 1)

    # Create an image grid
    grid = make_grid(x, nrow=10, padding=2)
    grid = F.to_pil_image(grid)
    size = [2 * px for px in grid.size]
    grid.resize(size, resample=Image.Resampling.NEAREST)
    output_dir = 'mnist-output/output'
    if ema:
        output_dir += "-ema"
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"-{epoch:04d}" if epoch is not None else ""
    filename = f"output{suffix}.png"
    grid.save(os.path.join(output_dir, filename))
    # print(f"Image saved as {filename}")
    torch.seed() # Reset seed

def test(image_channels=3,
         model_channels=32,
         activation_fn=torch.nn.SiLU,
         num_res_blocks=2,
         channel_mult=(1, 2, 2, 2),
         dropout=0.1,
         attention_resolutions=(2,),
         gpu=None,
         model_path='model.pth'):
    device = torch.device(f'cuda:{gpu}' if gpu is not None and torch.cuda.is_available() else 'cpu')
    noise_scheduler = NoiseScheduler().to(device)
    model = Model(image_channels=image_channels,
                  model_channels=model_channels,
                  activation_fn=activation_fn,
                  num_res_blocks=num_res_blocks,
                  channel_mult=channel_mult,
                  dropout=dropout,
                  attention_resolutions=attention_resolutions)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    _test(device, noise_scheduler, model, progress=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Diffusion Process with Configurable Parameters")
    parser.add_argument('command', choices=['train', 'test', 'eval'], help="Command to execute")
    parser.add_argument('--batch-size', type=int, default=128, help="Batch size")
    parser.add_argument('--epochs', type=int, default=2000, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--image-channels', type=int, default=1, help="Number of image channels")
    parser.add_argument('--model-channels', type=int, default=32, help="Number of channels in the model")
    parser.add_argument('--activation', type=str, default='silu', choices=ACTIVATION_FUNCTIONS.keys(), help="Activation function")
    parser.add_argument('--num-res-blocks', type=int, default=1, help="Number of residual blocks")
    parser.add_argument('--channel-mult', type=int, nargs=4, default=(1, 1, 2), help="Channel multipliers")
    # parser.add_argument('--hflip', action='store_true', help="Use horizontal flips")
    # parser.add_argument('--no-hflip', dest='hflip', action='store_false', help="Do not use horizontal flips")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate")
    parser.add_argument('--attention-resolutions', type=int, nargs='+', default=(2,), help="Resolutions to apply attention")
    parser.add_argument('--gpu', type=int, default=None, help="GPU index")
    parser.add_argument('--model', type=str, default='model.pth', help="Model file")
    parser.add_argument('--save-checkpoints', action='store_true', help="Save model checkpoints")
    args = parser.parse_args()

    activation_fn = ACTIVATION_FUNCTIONS[args.activation]

    if args.command == 'train':
        args.hflip = False
        train(batch_size=args.batch_size,
              epochs=args.epochs,
              lr=args.lr,
              image_channels=args.image_channels,
              model_channels=args.model_channels,
              activation_fn=activation_fn,
              num_res_blocks=args.num_res_blocks,
              channel_mult=args.channel_mult,
              hflip=args.hflip,
              dropout=args.dropout,
              attention_resolutions=args.attention_resolutions,
              gpu=args.gpu,
              save_checkpoints=args.save_checkpoints)
    elif args.command == 'test':
        test(model_channels=args.model_channels,
             activation_fn=activation_fn,
             num_res_blocks=args.num_res_blocks,
             channel_mult=args.channel_mult,
             dropout=args.dropout,
             attention_resolutions=args.attention_resolutions,
             gpu=args.gpu,
             model_path=args.model)