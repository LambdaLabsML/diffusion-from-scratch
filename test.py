import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from main import NoiseScheduler, Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

scheduler = NoiseScheduler().to(device)

model = Model().to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()

z = torch.randn(64, 1, 28, 28, device=device)
t_list = torch.arange(scheduler.steps-1, 0, -1, dtype=torch.long, device=device).unsqueeze(1).expand(-1, z.shape[0])
print(t_list.shape)
for t in t_list:
    pred_noise = model(z, t)
    z, x0 = scheduler.sample_prev_step(z, pred_noise, t)

z = (z + 1.) / 2.

# Move images to CPU for visualization if necessary
images = z.cpu()

# Create an image grid
grid = make_grid(images, nrow=8, padding=2)
grid = to_pil_image(grid)
grid.show()