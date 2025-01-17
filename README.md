# Diffusion from Scratch

This repo contains a guide to implementing diffusion models from scratch.

## Chapters

1. [Forward Process](01-forward-process/README.md)

   We begin by building a simple forward process that adds Gaussian noise to an image.
This is how we'll generate our input data for training our diffusion models.

2. [Reverse Process / Training](02-reverse-process/README.md)

   We implement the loss function, and training loop, and train a very simple model.

3. [Sampling / Inference](03-sampling/README.md)

   We show how to sample from our trained model to generate images.

4. [MNIST](04-mnist/README.md)

   We train a model on the MNIST dataset.
5. [CIFAR10](05-cifar10/README.md)

   We train a model on the CIFAR10 dataset.

6. [Conditional Models](06-conditional/README.md)

   We show how to train conditional models, allowing us to generate samples from a specific class.
7. [CelebA](07-celeba/README.md)

   We train a model on the CelebA dataset.