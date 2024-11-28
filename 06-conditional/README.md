# Condition Models

Let's now train some conditional models.

Instead of our model just taking $x_t$ and $t$ as input, we'll also pass the class index $c$ as input. This will allow us to generate samples from a specific class.

We'll use an additional embedding layer to encode the class index $c$.

```python
self.class_emb = torch.nn.Embedding(num_classes, embed_dim)
```

In our forward pass, we'll add the class embedding to the time embedding:

```python