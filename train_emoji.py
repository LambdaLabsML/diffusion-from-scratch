import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from emojis import EmojiDataset
from main import NoiseScheduler, Model, normalize
from unet import UNet, UNet2

class SingleImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_file, length=1000, transform=None):
        self.image = Image.open(image_file).convert('L')
        self.length = length
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.transform(self.image), 0



device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

scheduler = NoiseScheduler().to(device)

transform = transforms.Compose([transforms.Resize((64, 64), InterpolationMode.LANCZOS), transforms.ToTensor(), normalize])
# transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])

dataset = EmojiDataset(root='data/emojis/128', transform=transform)
# dataset = SingleImageDataset('data/fractal.png', length=60000, transform=transform)
# dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)


dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

model = UNet2(c=3, f=64).to(device)
model.train()

num_epochs = 1_000_000 // len(dataset)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = torch.nn.MSELoss()

for epoch in range(num_epochs):
    loss_epoch = 0
    for x, _ in dataloader:
        x = x.to(device)
        optimizer.zero_grad()
        noise = torch.randn_like(x, device=device)
        t = torch.randint(0, scheduler.steps, (x.shape[0],), device=device)
        x_noise = scheduler.add_noise(x, noise, t)
        pred_noise = model(x_noise, t)
        loss = criterion(pred_noise, noise)
        loss_epoch += loss.item()
        loss.backward()
        optimizer.step()
    loss_epoch /= len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss_epoch}")

torch.save(model.state_dict(), f'model_emoji.pth')
print(f"Saved model_emoji.pth")

# model = UNet2(c=3, f=64).to(device)
# model.load_state_dict(torch.load('model_emoji.pth'))
model.eval()

z = torch.randn(256, 3, 64, 64, device=device)
t_list = torch.arange(scheduler.steps-1, -1, -1, dtype=torch.long, device=device).unsqueeze(1).expand(-1, z.shape[0])

for t in t_list:
    print(f"t: {t[0].item()}\r", end='')
    with torch.no_grad():
        pred_noise = model(z, t)
    z, x0 = scheduler.sample_prev_step(z, pred_noise, t)

x0 = (x0 + 1.) / 2.

# Move images to CPU for visualization if necessary
images = x0.cpu()

# Create an image grid
grid = make_grid(images, nrow=16, padding=2)
grid = to_pil_image(grid)
grid.save(f'grid_emoji.png')
print(f"Saved grid_emoji.png")