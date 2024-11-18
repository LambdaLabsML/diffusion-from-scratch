import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor

class MnistNormalizer(nn.Module):
    def __init__(self):
        super(MnistNormalizer, self).__init__()
        self.register_buffer('mean', torch.tensor([0.1307]))
        self.register_buffer('std', torch.tensor([0.3081]))

    def forward(self, x):
        return (x - self.mean) / self.std



class MnistClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(MnistClassifier, self).__init__()
        self.normalizer = MnistNormalizer()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 2048)
        self.fc2 = nn.Linear(2048, num_classes)
        self.dropout = nn.Dropout(0.8)

    def forward(self, x):
        x = self.normalizer(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MnistFeatureExtractor(MnistClassifier):
    def __init__(self, num_classes=10):
        super(MnistFeatureExtractor, self).__init__(num_classes)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        super(MnistFeatureExtractor, self).load_state_dict(state_dict, strict, assign)
        del self.fc2
        del self.dropout

    def forward(self, x):
        x = self.normalizer(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)  # Flatten the tensor
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
            torch.save(model.state_dict(), 'mnist_classifier.pth')
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
    model = MnistClassifier().to(device)
    transform = torchvision.transforms.Compose([
        transforms.RandomRotation(10),  # Rotation
        transforms.RandomAffine(0, shear=10),  # Shearing
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Shifting up and down
        transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),  # Zooming
        transforms.Resize((28, 28)),  # Rescale
        transforms.ToTensor()
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
    val_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    criterion = torch.nn.CrossEntropyLoss()
    callbacks = [
        SaveBestModel(),
        EarlyStopping(patience=50, delta=0.0001)
    ]

    num_epochs = 2000
    validate(0, model, val_data_loader, criterion, [], device)
    for epoch in range(1, num_epochs+1):
        train(epoch, model, data_loader, optimizer, criterion, device)
        val_loss = validate(epoch, model, val_data_loader, criterion, callbacks, device)
        prev_lr = optimizer.param_groups[0]['lr']
        # scheduler.step(val_loss)
        # if prev_lr != optimizer.param_groups[0]['lr']:
        #     print('[Scheduler]', f'Learning rate changed to {optimizer.param_groups[0]["lr"]:.6f}')

if __name__ == '__main__':
    main()