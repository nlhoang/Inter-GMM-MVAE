import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import m_elbo
from vae_vision import Encoder_vision, Decoder_vision
from vae_audio import Encoder_audio, Decoder_audio
from vae_tactile import Encoder_tactile, Decoder_tactile
from utils import visualize_ls
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
        batch_size_current = vision.size(0) if vision is not None else audio.size(
            0) if audio is not None else tactile.size(0)
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


def train(args, model, data_loader, optimizer, device='mps', mu_prior=None, var_prior=None, save_epoch=0, desc=None):
    model.train()
    loss_average = 0
    loss_vision = 0
    loss_audio = 0
    loss_tactile = 0
    means = torch.zeros(1, args.latent_dim)
    logvars = torch.zeros(1, args.latent_dim)
    for batch_idx, data in enumerate(data_loader):
        if desc == 'vision_tactile':
            vision = data[0].to(device)
            audio = None
            tactile = data[1].to(device)
        elif desc == 'audio_tactile':
            vision = None
            audio = data[0].to(device)
            tactile = data[1].to(device)
        elif desc == 'vision_audio':
            vision = data[0].to(device)
            audio = data[1].to(device)
            tactile = None
        else:
            vision = data[0].to(device)
            audio = data[1].to(device)
            tactile = data[2].to(device)
        vision_recon, audio_recon, tactile_recon, mu, logvar = model(vision=vision, audio=audio, tactile=tactile)
        loss_vision = 0
        loss_audio = 0
        loss_tactile = 0

        if mu_prior is not None:
            loss_joint, la, lb, lc = m_elbo(args=args, recon_vision=vision_recon, vision=vision,
                                recon_audio=audio_recon, audio=audio, recon_tactile=tactile_recon, tactile=tactile,
                                mu=mu, logvar=logvar, mu_prior=mu_prior, var_prior=var_prior,
                                variational_beta=args.variational_beta, dim=args.latent_dim)
        else:
            loss_joint, la, lb, lc = m_elbo(args=args, recon_vision=vision_recon, vision=vision,
                                recon_audio=audio_recon, audio=audio, recon_tactile=tactile_recon, tactile=tactile,
                                mu=mu, logvar=logvar, variational_beta=args.variational_beta)

        optimizer.zero_grad()
        loss_joint.backward()
        optimizer.step()
        loss_average += loss_joint.item()
        loss_vision += la
        loss_audio += lb
        loss_tactile += lc
        if save_epoch != 0:
            for mean in mu:
                mean = mean.cpu()
                means = torch.cat((means, mean.unsqueeze(0)), dim=0)
            for lv in logvar:
                lv = lv.cpu()
                logvars = torch.cat((logvars, lv.unsqueeze(0)), dim=0)
    loss_average /= len(data_loader.dataset)
    return loss_average, means[1:], logvars[1:], loss_vision, loss_audio, loss_tactile


def train_old(args, model, data_loader, optimizer, device='mps', mu_prior=None, var_prior=None, save_epoch=0, desc=None):
    model.train()
    loss_average = 0
    loss_vision = 0
    loss_audio = 0
    loss_tactile = 0
    means = torch.zeros(1, args.latent_dim)
    logvars = torch.zeros(1, args.latent_dim)
    for batch_idx, data in enumerate(data_loader):
        if desc == 'vision_tactile':
            vision = data[0].to(device)
            audio = None
            tactile = data[1].to(device)
        elif desc == 'audio_tactile':
            vision = None
            audio = data[0].to(device)
            tactile = data[1].to(device)
        elif desc == 'vision_audio':
            vision = data[0].to(device)
            audio = data[1].to(device)
            tactile = None
        else:
            vision = data[0].to(device)
            audio = data[1].to(device)
            tactile = data[2].to(device)
        vision_recon, audio_recon, tactile_recon, mu, logvar = model(vision=vision, audio=audio, tactile=tactile)
        loss_vision = 0
        loss_audio = 0
        loss_tactile = 0
        no_average = 1

        if mu_prior is not None:
            loss_joint, la, lb, lc = m_elbo(args=args, recon_vision=vision_recon, vision=vision,
                                recon_audio=audio_recon, audio=audio, recon_tactile=tactile_recon, tactile=tactile,
                                mu=mu, logvar=logvar, mu_prior=mu_prior, var_prior=var_prior,
                                variational_beta=args.variational_beta, dim=args.latent_dim)
            if vision is not None:
                vision_recon1, _, _, mu_vision, logvar_vision = model(vision=vision, audio=None, tactile=None)
                loss_vision, _, _, _ = m_elbo(args=args, recon_vision=vision_recon1, vision=vision,
                                     recon_audio=None, audio=None, recon_tactile=None, tactile=None,
                                     mu=mu_vision, logvar=logvar_vision, mu_prior=mu_prior, var_prior=var_prior,
                                     variational_beta=args.variational_beta, dim=args.latent_dim)
                no_average += 1
            if audio is not None:
                _, audio_recon1, _, mu_audio, logvar_audio = model(vision=None, audio=audio, tactile=None)
                loss_audio, _, _, _ = m_elbo(args=args, recon_vision=None, vision=None,
                                    recon_audio=audio_recon1, audio=audio, recon_tactile=None, tactile=None,
                                    mu=mu_audio, logvar=logvar_audio, mu_prior=mu_prior, var_prior=var_prior,
                                    variational_beta=args.variational_beta, dim=args.latent_dim)
                no_average += 1
            if tactile is not None:
                _, _, tactile_recon1, mu_tactile, logvar_tactile = model(vision=None, audio=None, tactile=tactile)
                loss_tactile, _, _, _ = m_elbo(args=args, recon_vision=None, vision=None,
                                      recon_audio=None, audio=None, recon_tactile=tactile_recon1, tactile=tactile,
                                      mu=mu_tactile, logvar=logvar_tactile, mu_prior=mu_prior, var_prior=var_prior,
                                      variational_beta=args.variational_beta, dim=args.latent_dim)
                no_average += 1
        else:
            loss_joint, la, lb, lc = m_elbo(args=args, recon_vision=vision_recon, vision=vision,
                                recon_audio=audio_recon, audio=audio, recon_tactile=tactile_recon, tactile=tactile,
                                mu=mu, logvar=logvar, variational_beta=args.variational_beta)
            if vision is not None:
                vision_recon1, _, _, mu_vision, logvar_vision = model(vision=vision, audio=None, tactile=None)
                loss_vision, _, _, _  = m_elbo(args=args, recon_vision=vision_recon1, vision=vision,
                                     recon_audio=None, audio=None, recon_tactile=None, tactile=None,
                                     mu=mu_vision, logvar=logvar_vision, variational_beta=args.variational_beta)
                no_average += 1
            if audio is not None:
                _, audio_recon1, _, mu_audio, logvar_audio = model(vision=None, audio=audio, tactile=None)
                loss_audio, _, _, _ = m_elbo(args=args, recon_vision=None, vision=None,
                                    recon_audio=audio_recon1, audio=audio, recon_tactile=None, tactile=None,
                                    mu=mu_audio, logvar=logvar_audio, variational_beta=args.variational_beta)
                no_average += 1
            if tactile is not None:
                _, _, tactile_recon1, mu_tactile, logvar_tactile = model(vision=None, audio=None, tactile=tactile)
                loss_tactile, _, _, _ = m_elbo(args=args, recon_vision=None, vision=None,
                                      recon_audio=None, audio=None, recon_tactile=tactile_recon1, tactile=tactile,
                                      mu=mu_tactile, logvar=logvar_tactile, variational_beta=args.variational_beta)
                no_average += 1

        loss = (loss_joint + loss_vision + loss_audio + loss_tactile) / no_average
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_average += loss.item()
        loss_vision += la
        loss_audio += lb
        loss_tactile += lc
        if save_epoch != 0:
            for mean in mu:
                mean = mean.cpu()
                means = torch.cat((means, mean.unsqueeze(0)), dim=0)
            for lv in logvar:
                lv = lv.cpu()
                logvars = torch.cat((logvars, lv.unsqueeze(0)), dim=0)
    loss_average /= len(data_loader.dataset)
    return loss_average, means[1:], logvars[1:], loss_vision, loss_audio, loss_tactile


def elbo(recon_vision, vision, recon_audio, audio, recon_tactile, tactile, mu, logvar, variational_beta=1.0):
    recon_loss_vision = 0
    recon_loss_audio = 0
    recon_loss_tactile = 0
    mse_loss = nn.MSELoss()

    if recon_vision is not None and vision is not None:
        recon_loss_vision = mse_loss(recon_vision, vision)
    if recon_audio is not None and audio is not None:
        recon_loss_audio = mse_loss(recon_audio, audio)
    if recon_tactile is not None and tactile is not None:
        recon_loss_tactile = mse_loss(recon_tactile, tactile)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    kld = torch.mean(kl_divergence)
    loss = torch.mean(recon_loss_vision
                      + recon_loss_audio
                      + recon_loss_tactile
                      + variational_beta * kld)
    return loss, recon_loss_vision, recon_loss_audio, recon_loss_tactile


def train_original(model, data_loader, optimizer, device='mps'):
    model.train()
    loss_average = 0
    loss_vision = 0
    loss_audio = 0
    loss_tactile = 0
    for batch_idx, data in enumerate(data_loader):
        vision = data[0].to(device)
        audio = data[1].to(device)
        tactile = data[2].to(device)
        vision_recon, audio_recon, tactile_recon, mu, logvar = model(vision=vision, audio=audio, tactile=tactile)
        loss_vision = 0
        loss_audio = 0
        loss_tactile = 0
        no_average = 1
        loss, la, lb, lc = elbo(recon_vision=vision_recon, vision=vision, recon_audio=audio_recon, audio=audio,
                                      recon_tactile=tactile_recon, tactile=tactile, mu=mu, logvar=logvar)

        #loss = (loss_joint + loss_vision + loss_audio + loss_tactile) / no_average
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_average += loss.item()
        loss_vision += la
        loss_audio += lb
        loss_tactile += lc
    loss_average /= len(data_loader.dataset)
    print(loss_average, loss_vision, loss_audio, loss_tactile)


def latent_evaluate(model, data_loader, device='cpu'):
    model.eval()
    model.to(device)
    allmu = []

    for batch_idx, data in enumerate(data_loader):
        vision = data[0].to(device)
        audio = data[1].to(device)
        tactile = data[2].to(device)
        _, _, _, mu, _ = model(vision=vision, audio=audio, tactile=tactile)
        allmu.append(mu.detach().cpu().numpy())
    flattened_list = [item for sublist in allmu for item in sublist]
    visualize_ls(flattened_list, data_index, label_tactile5, '', 'MVAE')


data_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
              29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
              55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
              81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
              106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
              127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,
              148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164]
label_tactile = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
label_tactile5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

'''
import dataloader
batch_size = 165
device = 'cpu'
train_dataloader = dataloader.get_modalities(batch_size=batch_size, _vision=True, _audio=True, _tactile=True, shuffle=False)
model = MultiVAE(latent_dim=20, device=device)
name = 'MVAE.pth'
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=1e-5, eps=1e-3)
#for epoch in range(100):
#    train_original(model=model, data_loader=train_dataloader, optimizer=optimizer, device=device)
#torch.save(model.state_dict(), name)

model.load_state_dict(torch.load(name))
latent_evaluate(model, train_dataloader, device)
'''