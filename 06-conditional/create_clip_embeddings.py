import math
import os

from tqdm import tqdm

import clip
import torch
from torchvision.datasets import CIFAR10

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
dataset = CIFAR10(root="data", download=True, train=True, transform=preprocess)

# Create a DataLoader for batching
batch_size = 128  # Adjust based on available GPU memory
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

clip_embeddings = []
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for images, _ in tqdm(dataloader, total=math.ceil(len(dataset) / batch_size), desc="Computing CLIP embeddings"):
        images = images.to(device)  # Move batch to the device
        image_features = model.encode_image(images)  # Compute embeddings
        image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize embeddings
        clip_embeddings.append(image_features.cpu())  # Collect CPU tensors

# Save the embeddings
clip_embeddings = torch.cat(clip_embeddings)
torch.save(clip_embeddings, "cifar10.pt")

class CIFAR10Clip(CIFAR10):
    def __init__(self, clip_embeddings="cifar10.pt", **kwargs):
        super().__init__(**kwargs)
        self.clip_embeddings = torch.load(clip_embeddings, weights_only=True)

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image, label, self.clip_embeddings[index]

dataset = CIFAR10Clip(root="data", download=True, train=True)
image, label, clip_embedding = dataset[3637]
image.show()
print(f'Label: {dataset.classes[label]}')
print(f'Shape of the clip embedding: {clip_embedding.shape}')

clip_embedding = clip_embedding.to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in dataset.classes]).to(device)

# Calculate features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)

# Pick the top 10 most similar labels for the image
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * clip_embedding @ text_features.T).softmax(dim=-1)
print(f'Shape of the similarity tensor: {similarity.shape}')
values, indices = similarity.sort(descending=True)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{dataset.classes[index]:>16s}: {100 * value.item():.2f}%")