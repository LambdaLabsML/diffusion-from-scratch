import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from overrides import overrides
from torchvision import transforms
from torchvision.transforms import ToTensor

class Cifar10Normalizer(nn.Module):
    def __init__(self):
        super(Cifar10Normalizer, self).__init__()
        self.register_buffer('mean', torch.tensor([0.4914, 0.4822, 0.4465]))
        self.register_buffer('std', torch.tensor([0.2470, 0.2435, 0.2616]))

    def forward(self, x):
        return (x - self.mean.view(1, 3, 1, 1)) / self.std.view(1, 3, 1, 1)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        h = F.relu(self.norm1(x))
        h = self.conv1(h)
        h = F.relu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + (self.shortcut(x) if self.shortcut is not None else x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        h = self.norm(x)
        B, C, H, W = h.shape  # Reshape from (B, C, H, W) to (B, H * W, C) for multi-head attention
        h = h.view(B, C, H * W).permute(0, 2, 1)  # Shape: (B, H*W, C)
        h, _ = self.attention(h, h, h)  # Self-attention with keys, queries, values as `h`
        h = h.permute(0, 2, 1).view(B, C, H, W)  # Reshape back to (B, C, H, W)
        h = self.proj_out(h)  # Apply final projection
        return x + h


class Cifar10Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(Cifar10Classifier, self).__init__()
        self.normalizer = Cifar10Normalizer()
        ch = 64
        dropout = 0.1
        self.in_conv = nn.Conv2d(3, ch, kernel_size=3, padding=1)
        self.res1 = ResnetBlock(ch, ch, dropout=0.25)
        self.res2 = ResnetBlock(ch, ch, dropout=0.25)
        self.down1 = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)
        self.res3 = ResnetBlock(ch, ch*2, dropout=0.25)
        self.res4 = ResnetBlock(ch*2, ch*2, dropout=0.25)
        self.attn = AttentionBlock(ch*2)
        self.down2 = nn.Conv2d(ch*2, ch*2, kernel_size=3, stride=2, padding=1)
        self.res5 = ResnetBlock(ch*2, ch*2, dropout=0.25)
        self.res6 = ResnetBlock(ch*2, ch*2, dropout=0.25)
        self.down3 = nn.Conv2d(ch*2, ch*2, kernel_size=3, stride=2, padding=1)
        self.res7 = ResnetBlock(ch*2, ch*2, dropout=0.25)
        self.res8 = ResnetBlock(ch*2, ch*2, dropout=0.25)
        self.norm_out = nn.GroupNorm(32, ch*2)
        self.fc1 = nn.Linear(ch*2 * 4 * 4, 2048)
        self.dropout = nn.Dropout(0.8)
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.normalizer(x)
        x = self.in_conv(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.down1(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.attn(x)
        x = self.down2(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.down3(x)
        x = self.res7(x)
        x = self.res8(x)
        x = F.relu(self.norm_out(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Cifar10FeatureExtractor(Cifar10Classifier):
    def __init__(self, num_classes=10):
        super(Cifar10FeatureExtractor, self).__init__(num_classes)

    @overrides
    def load_state_dict(self, state_dict, strict=True, assign=False):
        super(Cifar10FeatureExtractor, self).load_state_dict(state_dict, strict, assign)
        del self.dropout
        del self.fc2

    def forward(self, x):
        x = self.normalizer(x)
        x = self.in_conv(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.down1(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.attn(x)
        x = self.down2(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.down3(x)
        x = self.res7(x)
        x = self.res8(x)
        x = F.relu(self.norm_out(x))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

class SaveBestModel:
    def __init__(self):
        self.best = None

    def __call__(self, epoch, val_loss, val_acc, model):
        if self.best is None:
            self.best = {'epoch': epoch, 'val_loss': val_loss, 'val_acc': val_acc}
        elif val_loss < self.best['val_loss']:
            self.best = {'epoch': epoch, 'val_loss': val_loss, 'val_acc': val_acc}
            torch.save(model.state_dict(), 'cifar10_classifier.pth')
            print('[SaveBestModel]', f'Best Epoch: {self.best["epoch"]}, Validation Loss: {self.best["val_loss"]:.4f}, Accuracy: {self.best["val_acc"]:.2f}%')

class EarlyStopping:
    def __init__(self, patience=10, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.early_stop = False
        self.best = None

    def __call__(self, epoch, val_loss, val_acc, model):
        if self.best is None:
            self.best = {'epoch': epoch, 'val_loss': val_loss, 'val_acc': val_acc}
        elif val_loss < (self.best['val_loss'] - self.delta):
            self.counter = 0
            self.best = {'epoch': epoch, 'val_loss': val_loss, 'val_acc': val_acc}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print('[EarlyStopping]', 'Stopping early')
                print('[EarlyStopping]', f'Best Epoch: {self.best["epoch"]}, Validation Loss: {self.best["val_loss"]:.4f}, Accuracy: {self.best["val_acc"]:.2f}%')
                sys.exit()
            print('[EarlyStopping]', f'Patience {self.counter}/{self.patience}')

def train(epoch, model, data_loader, optimizer, criterion, device):
    model.train()
    loss_epoch = 0
    for data, labels in data_loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
    loss_epoch /= len(data_loader)
    print(f'Epoch {epoch:04d}, Loss: {loss_epoch:.4f}')
    return loss_epoch


def validate(epoch, model, val_data_loader, criterion, callbacks, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        loss = 0
        total = 0
        for images, labels in val_data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels) * labels.size(0)
        accuracy = 100 * correct / total
        loss /= total
        print(f'Epoch {epoch:04d}, Validation Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%')
        for callback in callbacks:
            callback(epoch, loss, accuracy, model)
    return loss

def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = Cifar10Classifier().to(device)
    transform = torchvision.transforms.Compose([
        # transforms.RandomRotation(10),  # Rotation
        transforms.RandomAffine(0, shear=10),  # Shearing
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Shifting up and down
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),  # Zooming
        transforms.Resize((32, 32)),  # Rescale
        transforms.ToTensor()
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=ToTensor())
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=100)
    criterion = torch.nn.CrossEntropyLoss()
    callbacks = [
        SaveBestModel(),
        # EarlyStopping(patience=50, delta=0.0001)
    ]

    num_epochs = 2000
    validate(0, model, val_data_loader, criterion, [], device)
    for epoch in range(1, num_epochs+1):
        train_loss = train(epoch, model, data_loader, optimizer, criterion, device)
        val_loss = validate(epoch, model, val_data_loader, criterion, callbacks, device)
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(train_loss)
        if prev_lr != optimizer.param_groups[0]['lr']:
            print('[Scheduler]', f'Learning rate changed to {optimizer.param_groups[0]["lr"]:.6f}')

if __name__ == '__main__':
    main()