import torch
import torchvision
from torchvision import transforms

from main import NoiseScheduler, Model, normalize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

scheduler = NoiseScheduler().to(device)
transform = transforms.Compose([transforms.ToTensor(), normalize])
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
model = Model().to(device)
model.train()

num_epochs = 10
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
criterion = torch.nn.MSELoss()

for epoch in range(num_epochs):
    loss_epoch = 0
    for x, t in dataloader:
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

torch.save(model.state_dict(), 'model.pth')