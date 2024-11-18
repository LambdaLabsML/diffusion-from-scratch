import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Model parameters
        self.in_channels = 3
        self.out_channels = 3
        self.ch = 128
        self.dropout = 0.1

        # Timestep embedding
        self.temb_dense = nn.Sequential(
            nn.Linear(self.ch, self.ch * 4),
            nn.SiLU(),
            nn.Linear(self.ch * 4, self.ch * 4)
        )

        # Initial convolution
        self.in_conv = nn.Conv2d(self.in_channels, self.ch, kernel_size=3, padding=1)

        # Downsampling path (4 downsampling levels)
        self.down_block1 = ResnetBlock(self.ch, self.ch, self.ch * 4, dropout=self.dropout)
        self.down_block2 = ResnetBlock(self.ch, self.ch * 2, self.ch * 4, dropout=self.dropout)
        self.down_attn2 = AttentionBlock(self.ch * 2)
        self.downsample1 = Downsample(self.ch * 2, with_conv=True)

        self.down_block3 = ResnetBlock(self.ch * 2, self.ch * 2, self.ch * 4, dropout=self.dropout)
        self.down_block4 = ResnetBlock(self.ch * 2, self.ch * 4, self.ch * 4, dropout=self.dropout)
        self.downsample2 = Downsample(self.ch * 4, with_conv=True)

        self.down_block5 = ResnetBlock(self.ch * 4, self.ch * 4, self.ch * 4, dropout=self.dropout)
        self.down_block6 = ResnetBlock(self.ch * 4, self.ch * 4, self.ch * 4, dropout=self.dropout)
        self.down_attn3 = AttentionBlock(self.ch * 4)
        self.downsample3 = Downsample(self.ch * 4, with_conv=True)

        # Middle block
        self.mid_block1 = ResnetBlock(self.ch * 4, self.ch * 4, self.ch * 4, dropout=self.dropout)
        self.mid_attn = AttentionBlock(self.ch * 4)
        self.mid_block2 = ResnetBlock(self.ch * 4, self.ch * 4, self.ch * 4, dropout=self.dropout)

        # Upsampling path (mirroring downsampling)
        self.upsample1 = Upsample(self.ch * 4, with_conv=True)
        self.up_block1 = ResnetBlock(self.ch * 8, self.ch * 4, self.ch * 4, dropout=self.dropout)
        self.up_block2 = ResnetBlock(self.ch * 4, self.ch * 4, self.ch * 4, dropout=self.dropout)
        self.up_attn1 = AttentionBlock(self.ch * 4)

        self.upsample2 = Upsample(self.ch * 4, with_conv=True)
        self.up_block3 = ResnetBlock(self.ch * 8, self.ch * 4, self.ch * 4, dropout=self.dropout)
        self.up_block4 = ResnetBlock(self.ch * 4, self.ch * 2, self.ch * 4, dropout=self.dropout)

        self.upsample3 = Upsample(self.ch * 2, with_conv=True)
        self.up_block5 = ResnetBlock(self.ch * 4, self.ch * 2, self.ch * 4, dropout=self.dropout)
        self.up_block6 = ResnetBlock(self.ch * 2, self.ch, self.ch * 4, dropout=self.dropout)
        self.up_attn2 = AttentionBlock(self.ch * 2)

        # Output layer
        self.norm_out = nn.GroupNorm(32, self.ch)
        self.conv_out = nn.Conv2d(self.ch, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        # Timestep embedding
        temb = self.temb_dense(t)

        # Initial convolution
        h = self.in_conv(x)
        hs = [h]

        # Downsampling path
        h = self.down_block1(h, temb)
        h = self.down_block2(h, temb)
        h = self.down_attn2(h)
        hs.append(h)
        h = self.downsample1(h)

        h = self.down_block3(h, temb)
        h = self.down_block4(h, temb)
        hs.append(h)
        h = self.downsample2(h)

        h = self.down_block5(h, temb)
        h = self.down_block6(h, temb)
        h = self.down_attn3(h)
        hs.append(h)
        h = self.downsample3(h)

        # Middle block
        h = self.mid_block1(h, temb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb)

        # Upsampling path (using skip connections)
        h = self.upsample1(h)
        h = torch.cat([h, hs.pop()], dim=1)
        h = self.up_block1(h, temb)
        h = self.up_block2(h, temb)
        h = self.up_attn1(h)

        h = self.upsample2(h)
        h = torch.cat([h, hs.pop()], dim=1)
        h = self.up_block3(h, temb)
        h = self.up_block4(h, temb)

        h = self.upsample3(h)
        h = torch.cat([h, hs.pop()], dim=1)
        h = self.up_block5(h, temb)
        h = self.up_block6(h, temb)
        h = self.up_attn2(h)

        # Output layer
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
        self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        # Normalize and reshape for attention
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)

        # Apply multi-head attention
        h, _ = self.attention(h, h, h)

        # Reshape back to (B, C, H, W)
        h = h.permute(0, 2, 1).view(B, C, H, W)
        return x + self.proj_out(h)


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
