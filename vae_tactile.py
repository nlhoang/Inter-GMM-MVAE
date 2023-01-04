import torch
import torch.nn as nn
from utils import elbo

tactile_datasize = 300 * 32
const = 1e-6


class Encoder_tactile(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder_tactile, self).__init__()
        self.flatten = nn.Flatten()
        self.enc = nn.Sequential(
            nn.Linear(in_features=tactile_datasize, out_features=1024),
            nn.ReLU(True),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(True),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(True),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(True),
        )
        self.fc1 = nn.Linear(in_features=128, out_features=latent_dim)
        self.fc2 = nn.Linear(in_features=128, out_features=latent_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.enc(x)
        mu = self.fc1(x) + const
        logvar = self.fc2(x) + const
        return mu, logvar


class Decoder_tactile(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder_tactile, self).__init__()
        self.dec = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=128),
            nn.ReLU(True),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(True),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(True),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(True),
            nn.Linear(in_features=1024, out_features=tactile_datasize),
            nn.ReLU(True),
        )

    def forward(self, z):
        z = self.dec(z)
        #z = torch.sigmoid(z)
        z = z.view(-1, 300, 32)
        return z


class VAE(nn.Module):
    def __init__(self, latent_dim, device):
        super(VAE, self).__init__()
        self.encoder = Encoder_tactile(latent_dim)
        self.decoder = Decoder_tactile(latent_dim)
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
    num_batch = 0
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
        num_batch += 1
        if save_epoch != 0:
            for mean in mu:
                mean = mean.cpu()
                means = torch.cat((means, mean.unsqueeze(0)), dim=0)
            for lv in logvar:
                lv = lv.cpu()
                logvars = torch.cat((logvars, lv.unsqueeze(0)), dim=0)
    loss_average /= num_batch
    return loss_average, means[1:], logvars[1:]
