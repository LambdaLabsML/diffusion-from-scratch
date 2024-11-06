import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from main import NoiseScheduler, Model
from unet import UNet, UNet2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda:1')

scheduler = NoiseScheduler().to(device)

# model = Model().to(device)
# model = UNet(c=1, f=16).to(device)
model = UNet2(c=3, f=64).to(device)
model.load_state_dict(torch.load('hp_model.pth'))
model.eval()

# z = torch.randn(64, 1, 32, 32, device=device)
# z = torch.randn(64, 1, 28, 28, device=device)
z = torch.randn(1024, 3, 32, 32, device=device)
t_list = torch.arange(scheduler.steps-1, -1, -1, dtype=torch.long, device=device).unsqueeze(1).expand(-1, z.shape[0])
for t in t_list:
    with torch.no_grad():
        pred_noise = model(z, t)
    z, x0 = scheduler.sample_prev_step(z, pred_noise, t)

x0 = (x0 + 1.) / 2.

# Move images to CPU for visualization if necessary
images = x0.cpu()

# Create an image grid
grid = make_grid(images, nrow=32, padding=2)
grid = to_pil_image(grid)
grid.show()
