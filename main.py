import torch
import torch.nn.functional as F

class NoiseScheduler(torch.nn.Module):
    def __init__(self, steps=1000, beta_start=1e-4, beta_end=0.02):
        super(NoiseScheduler, self).__init__()
        self.steps = steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        beta = torch.linspace(beta_start, beta_end, steps)
        alpha = 1. - beta
        sqrt_alpha = torch.sqrt(alpha)
        alpha_cumprod = torch.cumprod(alpha, 0)
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1. - alpha_cumprod)

        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('sqrt_alpha', sqrt_alpha)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        self.register_buffer('sqrt_alpha_cumprod', sqrt_alpha_cumprod)
        self.register_buffer('sqrt_one_minus_alpha_cumprod', sqrt_one_minus_alpha_cumprod)


    def add_noise(self, x, noise, t):
        return (self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1) * x
                + self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1) * noise)

    def sample_prev_step(self, xt, pred_noise, t):
        x0 = ((xt - self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1) * pred_noise)
              / self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1))
        x0 = torch.clamp(x0, -1., 1.)

        mean = (xt - (self.beta[t].view(-1, 1, 1, 1) * pred_noise)
                / self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1))
        mean = mean / self.sqrt_alpha[t].view(-1, 1, 1, 1)

        var = ((1 - self.alpha[t].view(-1, 1, 1, 1)) * self.beta[t].view(-1, 1, 1, 1)
               / (1 - self.alpha_cumprod[t - 1].view(-1, 1, 1, 1)))
        sigma = torch.sqrt(var)
        z = torch.randn_like(xt)

        x = mean + sigma * z
        zero_t_mask = (t == 0)
        if zero_t_mask.any():
            x[zero_t_mask] = mean[zero_t_mask]

        return x, x0

class Model(torch.nn.Module):
    def __init__(self, c=1, t_steps=1000, t_emb_dim=128):
        super(Model, self).__init__()
        # self.embedding = torch.nn.Embedding(t_steps, t_emb_dim)
        self.embedding = torch.nn.Sequential(
            torch.nn.Embedding(t_steps, t_emb_dim),
            torch.nn.Linear(t_emb_dim, t_emb_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(t_emb_dim, t_emb_dim),
            torch.nn.SiLU(),
        )

        self.down1 = torch.nn.Conv2d(c, 16, kernel_size=3, stride=2, padding=1)
        self.down2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.up1 = torch.nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.up2 = torch.nn.ConvTranspose2d(16, c, kernel_size=4, stride=2, padding=1)

        self.embed1 = torch.nn.Linear(t_emb_dim, 16)
        self.embed2 = torch.nn.Linear(t_emb_dim, 32)
        self.embed3 = torch.nn.Linear(t_emb_dim, 16)

    def forward(self, x, t):
        # emb = F.silu(self.embedding(t))
        emb = self.embedding(t)

        x = self.down1(x)
        emb_proj = self.embed1(emb)
        x = x + emb_proj.view(-1, x.shape[1], 1, 1)

        x = F.relu(x)
        x = self.down2(x)
        emb_proj = self.embed2(emb)
        x = x + emb_proj.view(-1, x.shape[1], 1, 1)

        x = F.relu(x)
        x = self.up1(x)
        emb_proj = self.embed3(emb)
        x = x + emb_proj.view(-1, x.shape[1], 1, 1)

        x = F.relu(x)
        x = self.up2(x)
        return x

def normalize(x):
    return 2. * x - 1.