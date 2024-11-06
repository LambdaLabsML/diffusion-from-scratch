import torch
import torchvision
from PIL import Image
from ipykernel.pickleutil import class_type
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

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


class CifarSingleClassDataset(torch.utils.data.Dataset):
    def __init__(self, root, class_idx, transform=None):
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        self.class_idx = class_idx
        self.indices = [i for i, (_, label) in enumerate(self.dataset) if label == class_idx]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

scheduler = NoiseScheduler().to(device)

# transform = transforms.Compose([transforms.ToTensor(), normalize])
transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])

# dataset = SingleImageDataset('data/fractal.png', length=60000, transform=transform)
# dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

for class_idx in range(10):
    print('='*50)
    print(f"Training with class {class_idx}")
    dataset = CifarSingleClassDataset('./data', class_idx=class_idx, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    # model = Model().to(device)
    # model = UNet(c=1, f=16).to(device)
    model = UNet2(c=3, f=64).to(device)
    model.train()

    num_epochs = 200
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

    torch.save(model.state_dict(), f'model_{class_idx}.pth')
    print(f"Saved model_{class_idx}.pth")

    model.eval()

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
    grid.save(f'grid_{class_idx}.png')
    print(f"Saved grid_{class_idx}.png")