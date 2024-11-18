import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionModel(nn.Module):
    def __init__(self, in_channels, out_channels, ch, ch_mult, num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, ch, kernel_size=3, padding=1)
        self.temb_dense = nn.Sequential(
            nn.Linear(ch, ch * 4),
            nn.SiLU(),
            nn.Linear(ch * 4, ch * 4)
        )

        # Downsampling
        self.down_layers = nn.ModuleList()
        in_ch = ch
        for i_level, mult in enumerate(ch_mult):
            for _ in range(num_res_blocks):
                self.down_layers.append(ResnetBlock(in_ch, ch * mult, ch * 4, dropout=dropout))
                if in_ch in attn_resolutions:
                    self.down_layers.append(AttentionBlock(ch * mult))
            if i_level != len(ch_mult) - 1:
                self.down_layers.append(Downsample(ch * mult, with_conv=resamp_with_conv))
            in_ch = ch * mult

        # Middle
        self.mid_block1 = ResnetBlock(in_ch, in_ch, ch * 4, dropout=dropout)
        self.mid_attn = AttentionBlock(in_ch)
        self.mid_block2 = ResnetBlock(in_ch, in_ch, ch * 4, dropout=dropout)

        # Upsampling
        self.up_layers = nn.ModuleList()
        for i_level, mult in reversed(list(enumerate(ch_mult))):
            for _ in range(num_res_blocks + 1):
                self.up_layers.append(ResnetBlock(in_ch, ch * mult, ch * 4, dropout=dropout))
                if in_ch in attn_resolutions:
                    self.up_layers.append(AttentionBlock(ch * mult))
            if i_level != 0:
                self.up_layers.append(Upsample(ch * mult, with_conv=resamp_with_conv))
            in_ch = ch * mult

        # Output
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        temb = self.temb_dense(t)

        h = self.in_conv(x)
        hs = [h]
        for layer in self.down_layers:
            h = layer(h, temb) if isinstance(layer, ResnetBlock) else layer(h)
            hs.append(h)

        h = self.mid_block1(h, temb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb)

        for layer in self.up_layers:
            h = layer(torch.cat([h, hs.pop()], dim=1), temb) if isinstance(layer, ResnetBlock) else layer(h)

        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)
        return h


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x, temb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h += self.temb_proj(F.silu(temb)).view(-1, h.shape[1], 1, 1)
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + (self.shortcut(x) if self.shortcut is not None else x)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = torch.chunk(self.qkv(h), chunks=3, dim=1)

        q = q.view(q.shape[0], q.shape[1], -1)
        k = k.view(k.shape[0], k.shape[1], -1)
        v = v.view(v.shape[0], v.shape[1], -1)

        w = torch.einsum('bci,bcj->bij', q, k) * (x.shape[1] ** -0.5)
        w = torch.softmax(w, dim=-1)

        h = torch.einsum('bij,bcj->bci', w, v).view(x.shape)
        h = self.proj_out(h)
        return x + h


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x
