import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import m_elbo
from vae_vision import Encoder_vision, Decoder_vision
from vae_audio import Encoder_audio, Decoder_audio
from vae_tactile import Encoder_tactile, Decoder_tactile

eta = 1e-6
eps = 1e-8


class MultiVAE(nn.Module):
    def __init__(self, latent_dim, device):
        super(MultiVAE, self).__init__()
        self.encoder_vision = Encoder_vision(latent_dim)
        self.decoder_vision = Decoder_vision(latent_dim)
        self.encoder_audio = Encoder_audio(latent_dim)
        self.decoder_audio = Decoder_audio(latent_dim)
        self.encoder_tactile = Encoder_tactile(latent_dim)
        self.decoder_tactile = Decoder_tactile(latent_dim)
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, vision=None, audio=None, tactile=None):
        mu, logvar, mu_vision, logvar_vision, mu_audio, logvar_audio, mu_tactile, logvar_tactile = self.encoder(vision=vision, audio=audio, tactile=tactile)
        latent_vision = self.reparameterize(mu_vision, logvar_vision)
        latent_audio = self.reparameterize(mu_audio, logvar_audio)
        latent_tactile = self.reparameterize(mu_tactile, logvar_tactile)
        if vision is not None:
            vision_recon = self.decoder_vision(latent_vision)
        else:
            vision_recon = None
        if audio is not None:
            audio_recon = self.decoder_audio(latent_audio)
        else:
            audio_recon = None
        if tactile is not None:
            tactile_recon = self.decoder_tactile(latent_tactile)
        else:
            tactile_recon = None
        return vision_recon, audio_recon, tactile_recon, mu, logvar

    def encoder(self, vision=None, audio=None, tactile=None):
        if vision is not None:
            mu_vision, logvar_vision = self.encoder_vision(vision)
        else:
            mu_vision, logvar_vision = None, None
        if audio is not None:
            mu_audio, logvar_audio = self.encoder_audio(audio)
        else:
            mu_audio, logvar_audio = None, None
        if tactile is not None:
            mu_tactile, logvar_tactile = self.encoder_tactile(tactile)
        else:
            mu_tactile, logvar_tactile = None, None

        mu, logvar = mixture_3experts(self.latent_dim, mu_vision, logvar_vision, mu_audio, logvar_audio, mu_tactile, logvar_tactile)
        return mu, logvar, mu_vision, logvar_vision, mu_audio, logvar_audio, mu_tactile, logvar_tactile

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps_ = Variable(std.data.new(std.size()).normal_())
            return eps_.mul(std).add_(mu)
        else:
            return mu


def mixture_3experts(dim, mu1, logvar1, mu2, logvar2, mu3, logvar3):
    id1 = dim // 3
    id2 = 2 * dim // 3
    mu = torch.cat((mu1[:, 0:id1], mu2[:, id1:id2], mu3[:, id2:dim]), dim=1)
    logvar = torch.cat((logvar1[:, 0:id1], logvar2[:, id1:id2], logvar3[:, id2:dim]), dim=1)
    return mu, logvar


def mixture_2experts(dim, mu1, logvar1, mu2, logvar2):
    mu = torch.cat((mu1[:, 0:dim // 2], mu2[:, dim // 2:dim]), dim=1)
    logvar = torch.cat((logvar1[:, 0:dim // 2], logvar2[:, dim // 2:dim]), dim=1)
    return mu, logvar


def train(args, model, data_loader, optimizer, device='mps', mu_prior=None, var_prior=None, save_epoch=0):
    model.train()
    loss_average = 0
    num_batch = 0
    means = torch.zeros(1, args.latent_dim)
    logvars = torch.zeros(1, args.latent_dim)
    for batch_idx, data in enumerate(data_loader):
        vision = data[0].to(device)
        audio = data[1].to(device)
        tactile = data[2].to(device)
        vision_recon, audio_recon, tactile_recon, mu, logvar = model(vision=vision, audio=audio, tactile=tactile)

        if mu_prior is not None:
            batch_head = batch_idx * args.batch_size
            batch_end = batch_head + args.batch_size
            mu_prior_batch = mu_prior[batch_head:batch_end]
            var_prior_batch = var_prior[batch_head:batch_end]
            loss = m_elbo(args=args, recon_vision=vision_recon, vision=vision,
                          recon_audio=audio_recon, audio=audio, recon_tactile=tactile_recon, tactile=tactile,
                          mu=mu, logvar=logvar, mu_prior=mu_prior_batch, var_prior=var_prior_batch,
                          dim=args.latent_dim, variational_beta=args.variational_beta)
        else:
            loss = m_elbo(args=args, recon_vision=vision_recon, vision=vision,
                          recon_audio=audio_recon, audio=audio, recon_tactile=tactile_recon, tactile=tactile,
                          mu=mu, logvar=logvar, variational_beta=args.variational_beta)

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
