import torch

# from part_c_cifar_silu import Model
# from part_d_cifar_arch import Model
from part_h_cifar_attn import Model, PositionalEncoding

# Count number of parameters
model = Model(image_channels=3,
              model_channels=128,
              activation_fn=torch.nn.SiLU,
              num_res_blocks=2,
              channel_mult=(1, 2, 2, 2),
              dropout=0.1,
              attention_resolutions=(2,))
n_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {n_params:,}")

# Count number of parameters
model = Model(image_channels=1,
              model_channels=32,
              activation_fn=torch.nn.SiLU,
              num_res_blocks=1,
              channel_mult=(1, 1, 2),
              dropout=0.1,
              attention_resolutions=(2,))
n_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {n_params:,}")
x = torch.randn(1, 1, 28, 28)
print(x.shape)
t = torch.tensor((1,), dtype=torch.long)
x = model(x, t)
print(x.shape)

"""
128, 2, (1, 2, 2), (2,)
Number of parameters: 25,305,601

64, 2, (1, 2, 2), (2,)
6,336,769

32, 2, (1, 2, 2), (2,)
1,589,377

32, 1, (1, 2, 2), (2,)
996,769

32, 2, (1, 1, 2), (2,)
1,084,481

32, 1, (1, 1, 2), (2,)
669,889

32, 2, (1, 1, 1), (2,)
564,449
"""

"""
celebA
lr=2e-5
batch_size=64
dropout=0.0
hflip
model_channels=128
channel_mult=(1, 1, 2, 2, 4, 4)
num_res_blocks=2
attention_resolutions=(8,)
"""

model = Model(image_channels=3,
                model_channels=128,
                activation_fn=torch.nn.SiLU,
                num_res_blocks=2,
                channel_mult=(1, 1, 2, 2, 4, 4),
                dropout=0.0,
                attention_resolutions=(8,))
n_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {n_params:,}")

# Number of parameters: 108,417,027

import torch
import torch.nn as nn

class CelebAConditioning(nn.Module):
    def __init__(self, attr_dim, ch):
        super().__init__()
        # Attribute embedding size (split embedding space among attributes)
        embed_dim = ch * 4

        self.embed = nn.Sequential(
            nn.Embedding(attr_dim * 2, ch),
            nn.Flatten(start_dim=1),
            nn.Linear(attr_dim * ch, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, attr):
        return self.embed(attr)

class TimeEmbedding(nn.Module):
    def __init__(self, ch):
        super().__init__()
        embed_dim = ch * 4
        self.embed = torch.nn.Sequential(
            PositionalEncoding(ch),
            torch.nn.Linear(ch, embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.SiLU(),
        )

    def forward(self, t):
        return self.embed(t)

ch = 128
model = TimeEmbedding(ch)
n_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters for time embedding: {n_params:,}")

# Define the model
attr_dim = 40  # Number of binary attributes
embed_dim = 512  # Embedding dimension for diffusion model
model_channels = 128  # Number of channels in the diffusion model
model = CelebAConditioning(attr_dim, model_channels)
# Number of parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters for conditioning: {n_params:,}")

# Example input: batch of binary attributes
batch_size = 8
attributes = torch.randint(0, 2, (batch_size, attr_dim))  # Random binary attributes

# Forward pass
conditioning_vector = model(attributes)  # Shape: (batch_size, embed_dim)

print("Conditioning Vector Shape:", conditioning_vector.shape)


class CelebAConditioning(nn.Module):
    def __init__(self, attr_dim, ch, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        # Attribute embedding size (split embedding space among attributes)
        embed_dim = ch * 4

        self.embed = nn.Sequential(
            nn.Embedding(attr_dim * 3, ch),
            nn.Flatten(start_dim=1),
            nn.Linear(attr_dim * ch, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, attr, training=True):
        attr = attr + 1  # Shift attribute values to [0, 1, 2] (unknown, false, true)

        if training and self.dropout > 0:
            # Apply dropout to the attributes
            mask = torch.rand(attr.shape, device=attr.device) > self.dropout
            attr = attr * mask  # Mask out some attributes (set them to zero)

        return self.embed(attr)

model = CelebAConditioning(attr_dim, model_channels)
# Number of parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters for conditioning: {n_params:,}")

# Example input: batch of binary attributes
batch_size = 8
attributes = torch.randint(-1, 2, (batch_size, attr_dim))  # Random binary attributes

# Forward pass
conditioning_vector = model(attributes)  # Shape: (batch_size, embed_dim)

print("Conditioning Vector Shape:", conditioning_vector.shape)