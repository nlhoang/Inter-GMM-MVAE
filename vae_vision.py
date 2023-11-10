import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import elbo
import dataloader
from utils import visualize_ls
vision_dataSize = torch.Size([3, 256, 256])
vision_channel = vision_dataSize[0]
fBase = 8
const = 1e-6


class Encoder_vision(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder_vision, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(vision_channel, fBase, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            nn.Conv2d(fBase * 4, fBase * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            nn.Conv2d(fBase * 8, fBase * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True),
            nn.Conv2d(fBase * 16, fBase * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 32),
            nn.ReLU(True),
        )
        self.c1 = nn.Conv2d(fBase * 32, latent_dim, 4, 1, 0, bias=False)
        self.c2 = nn.Conv2d(fBase * 32, latent_dim, 4, 1, 0, bias=False)

    def forward(self, x):
        x = self.enc(x)
        mu = self.c1(x).squeeze() + const
        logvar = F.softplus(self.c2(x)).squeeze() + const
        return mu, logvar


class Decoder_vision(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder_vision, self).__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, fBase * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(fBase * 32),
            nn.ReLU(True),
            nn.ConvTranspose2d(fBase * 32, fBase * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(fBase * 16, fBase * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(fBase * 8, fBase * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(fBase * 4, fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            nn.ConvTranspose2d(fBase, vision_channel, 4, 2, 1, bias=False),
            nn.ReLU(True),
        )

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        z = self.dec(z.view(-1, *z.size()[-3:]))
        return z


class VAE(nn.Module):
    def __init__(self, latent_dim, device):
        super(VAE, self).__init__()
        self.encoder = Encoder_vision(latent_dim)
        self.decoder = Decoder_vision(latent_dim)
        self.device = device

    def forward(self, x):
        mu, logvar = self.encoder(x)
        latent = self.reparameterize(mu, logvar)
        recon = self.decoder(latent)
        return recon, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


def train(args, model, data_loader, optimizer, device='mps', mu_prior=None, var_prior=None, save_epoch=0):
    model.train()
    loss_average = 0
    means = torch.zeros(1, args.latent_dim)
    logvars = torch.zeros(1, args.latent_dim)
    for batch_idx, x in enumerate(data_loader):
        x = x.to(device)
        recon, mu, logvar = model(x)
        if mu_prior is not None:
            batch_head = batch_idx * args.batch_size
            batch_end = batch_head + args.batch_size
            mu_prior_batch = mu_prior[batch_head:batch_end]
            var_prior_batch = var_prior[batch_head:batch_end]
            loss = elbo(recon, x, mu, logvar, mu_prior=mu_prior_batch, var_prior=var_prior_batch, dim=args.latent_dim,
                        variational_beta=args.variational_beta)
        else:
            loss = elbo(recon, x, mu, logvar, mu_prior=mu_prior, var_prior=var_prior,
                        variational_beta=args.variational_beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_average += loss.item()
        if save_epoch != 0:
            for mean in mu:
                mean = mean.cpu()
                means = torch.cat((means, mean.unsqueeze(0)), dim=0)
            for lv in logvar:
                lv = lv.cpu()
                logvars = torch.cat((logvars, lv.unsqueeze(0)), dim=0)
    loss_average /= len(data_loader.dataset)
    return loss_average, means[1:], logvars[1:]


def elbo_original(recon, x, mu, logvar, beta=1.0):
    mse_loss = nn.MSELoss()
    recon_loss = mse_loss(recon, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    loss = torch.mean(recon_loss + beta * kl_divergence)
    return loss


def train_original(model, data_loader, epochs=100, beta=1, device='mps'):
    model.train()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=1e-5)
    model.to(device)
    for epoch in range(epochs):
        loss_average = 0
        for batch_idx, x in enumerate(data_loader):
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss = elbo_original(recon, x, mu, logvar, beta=beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_average += loss.item()
        loss_average /= len(data_loader.dataset)
        print('Epoch', epoch, ':', loss_average)
