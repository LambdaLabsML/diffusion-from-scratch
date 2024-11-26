import torch

# from part_c_cifar_silu import Model
# from part_d_cifar_arch import Model
from part_e_cifar_deeper import Model

model = Model(model_channels=32)
x = torch.randn(1, 3, 32, 32)
print(x.shape)
t = torch.tensor((1,), dtype=torch.long)
x = model(x, t)
print(x.shape)