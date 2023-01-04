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
        mu, logvar = self.encoder(vision=vision, audio=audio, tactile=tactile)
        latent = self.reparameterize(mu, logvar)
        if vision is not None:
            vision_recon = self.decoder_vision(latent)
        else:
            vision_recon = None
        if audio is not None:
            audio_recon = self.decoder_audio(latent)
        else:
            audio_recon = None
        if tactile is not None:
            tactile_recon = self.decoder_tactile(latent)
        else:
            tactile_recon = None
        return vision_recon, audio_recon, tactile_recon, mu, logvar

    def encoder(self, vision=None, audio=None, tactile=None, mu_prior=None, var_prior=None):
        batch_size_current = vision.size(0) if vision is not None else audio.size(0) if audio is not None else tactile.size(0)
        mu, logvar = self.prior_distribution(batch_size_current, mu_prior=mu_prior, var_prior=var_prior)
        mu = mu.to(self.device)
        logvar = logvar.to(self.device)
        if vision is not None:
            mu_vision, logvar_vision = self.encoder_vision(vision)
            mu = torch.cat((mu, mu_vision.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, logvar_vision.unsqueeze(0)), dim=0)
        if audio is not None:
            mu_audio, logvar_audio = self.encoder_audio(audio)
            mu = torch.cat((mu, mu_audio.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, logvar_audio.unsqueeze(0)), dim=0)
        if tactile is not None:
            mu_tactile, logvar_tactile = self.encoder_tactile(tactile)
            mu = torch.cat((mu, mu_tactile.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, logvar_tactile.unsqueeze(0)), dim=0)

        mu, logvar = product_experts(mu, logvar)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps_ = Variable(std.data.new(std.size()).normal_())
            return eps_.mul(std).add_(mu)
        else:
            return mu

    def prior_distribution(self, _batch_size, mu_prior=None, var_prior=None):
        if mu_prior is None:
            mu = Variable(torch.zeros(1, _batch_size, self.latent_dim))
            logvar = Variable(torch.zeros(1, _batch_size, self.latent_dim))
        else:
            mu = Variable(torch.from_numpy(mu_prior[None, :]).to(torch.float32))
            logvar = Variable(torch.from_numpy(var_prior[None, :]).to(torch.float32))
        return mu, logvar


def product_experts(mu, logvar):
    var = torch.exp(logvar) + eps
    t = 1. / (var + eps)
    pd_mu = torch.sum(mu * t, dim=0) / torch.sum(t, dim=0)
    pd_var = 1. / torch.sum(t, dim=0)
    pd_logvar = torch.log(pd_var + eps)
    return pd_mu, pd_logvar


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
        vision_recon1, _, _, mu_vision, logvar_vision = model(vision=vision, audio=None, tactile=None)
        _, audio_recon1, _, mu_audio, logvar_audio = model(vision=None, audio=audio, tactile=None)
        _, _, tactile_recon1, mu_tactile, logvar_tactile = model(vision=None, audio=None, tactile=tactile)

        if mu_prior is not None:
            batch_head = batch_idx * args.batch_size
            batch_end = batch_head + args.batch_size
            mu_prior_batch = mu_prior[batch_head:batch_end]
            var_prior_batch = var_prior[batch_head:batch_end]
            loss_joint = m_elbo(args=args, recon_vision=vision_recon, vision=vision,
                                recon_audio=audio_recon, audio=audio, recon_tactile=tactile_recon, tactile=tactile,
                                mu=mu, logvar=logvar, mu_prior=mu_prior_batch, var_prior=var_prior_batch,
                                variational_beta=args.variational_beta, dim=args.latent_dim)
            loss_vision = m_elbo(args=args, recon_vision=vision_recon, vision=vision,
                                 recon_audio=None, audio=None, recon_tactile=None, tactile=None,
                                 mu=mu, logvar=logvar, mu_prior=mu_prior_batch, var_prior=var_prior_batch,
                                 variational_beta=args.variational_beta, dim=args.latent_dim)
            loss_audio = m_elbo(args=args, recon_vision=None, vision=None,
                                recon_audio=audio_recon, audio=audio, recon_tactile=None, tactile=None,
                                mu=mu, logvar=logvar, mu_prior=mu_prior_batch, var_prior=var_prior_batch,
                                variational_beta=args.variational_beta, dim=args.latent_dim)
            loss_tactile = m_elbo(args=args, recon_vision=None, vision=None,
                                  recon_audio=None, audio=None, recon_tactile=tactile_recon, tactile=tactile,
                                  mu=mu, logvar=logvar, mu_prior=mu_prior_batch, var_prior=var_prior_batch,
                                  variational_beta=args.variational_beta, dim=args.latent_dim)
        else:
            loss_joint = m_elbo(args=args, recon_vision=vision_recon, vision=vision,
                                recon_audio=audio_recon, audio=audio, recon_tactile=tactile_recon, tactile=tactile,
                                mu=mu, logvar=logvar, variational_beta=args.variational_beta)
            loss_vision = m_elbo(args=args, recon_vision=vision_recon, vision=vision,
                                 recon_audio=None, audio=None, recon_tactile=None, tactile=None,
                                 mu=mu, logvar=logvar, variational_beta=args.variational_beta)
            loss_audio = m_elbo(args=args, recon_vision=None, vision=None,
                                recon_audio=audio_recon, audio=audio, recon_tactile=None, tactile=None,
                                mu=mu, logvar=logvar, variational_beta=args.variational_beta)
            loss_tactile = m_elbo(args=args, recon_vision=None, vision=None,
                                  recon_audio=None, audio=None, recon_tactile=tactile_recon, tactile=tactile,
                                  mu=mu, logvar=logvar, variational_beta=args.variational_beta)
        loss = (loss_joint + loss_vision + loss_audio + loss_tactile) / 4
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
