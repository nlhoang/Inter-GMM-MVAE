import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import elbo
import dataloader
from utils import visualize_ls
audio_datasize = 20 * 700
fBase = 8
const = 1e-6


class Encoder_audio(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder_audio, self).__init__()
        self.enc = nn.Sequential(
            # 20 x 700
            nn.Conv2d(1, fBase, (3, 6), (1, 4), (1, 1), bias=False),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # 20 x 175
            nn.Conv2d(fBase, fBase * 2, (5, 5), (1, 4), (0, 1), bias=False),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # 16 x 44
            nn.Conv2d(fBase * 2, fBase * 4, (4, 6), (2, 4), (1, 1), bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # 8 x 11
            nn.Conv2d(fBase * 4, fBase * 8, (4, 4), (2, 3), (1, 1), bias=False),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            # 4 x 4
        )
        self.c1 = nn.Conv2d(fBase * 8, latent_dim, (4, 4), (1, 1), (0, 0), bias=False)
        self.c2 = nn.Conv2d(fBase * 8, latent_dim, (4, 4), (1, 1), (0, 0), bias=False)

    def forward(self, x):
        x = self.enc(x)
        mu = self.c1(x).squeeze() + const
        logvar = F.softplus(self.c2(x)).squeeze() + const
        return mu, logvar


class Decoder_audio(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder_audio, self).__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, fBase * 8, (4, 4), (1, 1), (0, 0), bias=False),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            # 4 x 4
            nn.ConvTranspose2d(fBase * 8, fBase * 4, (4, 4), (2, 3), (1, 1), bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # 8 x 11
            nn.ConvTranspose2d(fBase * 4, fBase * 2, (4, 6), (2, 4), (1, 1), bias=False),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # 16 x 44
            nn.ConvTranspose2d(fBase * 2, fBase, (5, 5), (1, 4), (0, 1), bias=False),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # 20 x 175
            nn.ConvTranspose2d(fBase, 1, (3, 6), (1, 4), (1, 1), bias=False),
            nn.ReLU(True),
            # 20 x 700
        )

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        z = self.dec(z.view(-1, *z.size()[-3:]))
        return z


class VAE(nn.Module):
    def __init__(self, latent_dim, device):
        super(VAE, self).__init__()
        self.encoder = Encoder_audio(latent_dim)
        self.decoder = Decoder_audio(latent_dim)
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

