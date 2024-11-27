import os
import clip
import torch
from torchvision.datasets import CIFAR10

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
dataset = CIFAR10(root="data", download=True, train=True)

clip_embeddings = []
for idx, (image, label) in enumerate(dataset):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        clip_embeddings.append(image_features.cpu())

# Save the embeddings
clip_embeddings = torch.stack(clip_embeddings)
torch.save(clip_embeddings, "cifar10.pt")

class CIFAR10Clip(CIFAR10):
    def __init__(self, clip_embeddings="cifar10.pt", **kwargs):
        super().__init__(**kwargs)
        self.clip_embeddings = torch.load(clip_embeddings)

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image, label, self.clip_embeddings[index]

dataset = CIFAR10Clip(root="data", download=True, train=True)
image, label, clip_embedding = dataset[3637]
print(f'Label: {dataset.classes[label]}')
print(f'Shape of the clip embedding: {clip_embedding.shape}')

text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in dataset.classes]).to(device)

# Calculate features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)

# Pick the top 10 most similar labels for the image
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(10)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{dataset.classes[index]:>16s}: {100 * value.item():.2f}%")