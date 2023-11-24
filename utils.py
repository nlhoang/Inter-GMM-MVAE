import sys
import os
import shutil
import torch
from torch import nn
import csv
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pickle

colors = ['Green', 'Blue', 'Brown', 'Red', 'Black', 'Yellow', 'DarkSlateBlue',  'BurlyWood', 'Blue',
          'Chocolate', 'DarkBlue', 'BlueViolet', 'LightBlue', 'CadetBlue', 'Chartreuse', 'Coral', 'CornflowerBlue',
          'Cornsilk', 'AliceBlue', 'DarkBlue', 'DarkCyan', 'DarkGoldenRod', 'Beige', 'DarkSlateBlue', 'Orange',
          'DarkGreen', 'Chocolate', 'DarkMagenta', 'Orange','DarkOliveGreen',  'DarkOrchid', 'Purple', 'DarkRed',
          'DarkSalmon', 'DarkSeaGreen', 'DarkSlateBlue', 'Green', 'Blue', 'Brown', 'Red', 'Bisque']


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def save_checkpoint(state, is_best=False, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename), os.path.join(folder, 'model_best.pth.tar'))


def load_checkpoint(file_path, use_cuda=False):
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)
    return checkpoint


def param_count(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def visualize_ls(means, labels, save_dir, description):
    colors = ["red", "green", "blue", "orange", "purple", "yellow", "black", "cyan", '#a65628', '#f781bf']
    points_pca = PCA(n_components=2, random_state=0).fit_transform(means)
    points_tsne = TSNE(n_components=2, random_state=0).fit_transform(means)

    # TSNE
    plt.figure(figsize=(10, 10))
    for p, l in zip(points_tsne, labels):
        plt.title("TSNE", fontsize=24)
        plt.tick_params(labelsize=17)
        plt.scatter(p[0], p[1], marker="${}$".format(l), c=colors[l], s=100)
    plt.savefig(save_dir + 'tsne_' + description + '.png')
    plt.close()

    # PCA
    plt.figure(figsize=(10, 10))
    for p, l in zip(points_pca, labels):
        plt.title("PCA", fontsize=24)
        plt.tick_params(labelsize=17)
        plt.scatter(p[0], p[1], marker="${}$".format(l), c=colors[l], s=100)
    plt.savefig(save_dir + 'pca_' + description + '.png')
    plt.close()


def visualize_tsne(means, labels, save_dir, description):
    colors = ["red", "green", "blue", "orange", "purple", "yellow", "black", "cyan", '#a65628', '#f781bf']
    points_tsne = TSNE(n_components=2, random_state=0).fit_transform(means)

    # TSNE
    plt.figure(figsize=(10, 10))
    for p, l in zip(points_tsne, labels):
        plt.title("TSNE", fontsize=24)
        plt.tick_params(labelsize=17)
        plt.scatter(p[0], p[1], marker="${}$".format(l), c=colors[l], s=100)
    plt.savefig(save_dir + 'tsne_' + description + '.png')
    plt.close()


def visualize_pca(means, labels, save_dir, description):
    colors = ["red", "green", "blue", "orange", "purple", "yellow", "black", "cyan", '#a65628', '#f781bf']
    points_pca = PCA(n_components=2, random_state=0).fit_transform(means)

    # PCA
    plt.figure(figsize=(10, 10))
    for p, l in zip(points_pca, labels):
        plt.title("PCA", fontsize=24)
        plt.tick_params(labelsize=17)
        plt.scatter(p[0], p[1], marker="${}$".format(l), c=colors[l], s=100)
    plt.savefig(save_dir + 'pca_' + description + '.png')
    plt.close()


def save_toFile(path, file_name, data_saved, rows=0):
    f = open(path + file_name, 'w')
    writer = csv.writer(f)
    if rows == 0:
        writer.writerow(data_saved)
    if rows == 1:
        writer.writerows(data_saved)
    f.close()


def elbo(recon, x, mu, logvar, mu_prior=None, var_prior=None, dim=None, variational_beta=1.0):
    mse_loss = nn.MSELoss()
    recon_loss = mse_loss(recon, x)
    if mu_prior is None:
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    else:
        kl_divergence = kullback_leibler_divergence(dim, mu_prior, var_prior, mu, logvar)
    tensor = torch.masked_fill(kl_divergence, torch.isinf(kl_divergence) | torch.isnan(kl_divergence), 0.0)
    kld = torch.mean(tensor)
    loss = torch.mean(recon_loss + variational_beta * kld)
    return loss


def m_elbo(args, recon_vision, vision, recon_audio, audio, recon_tactile, tactile,
           mu, logvar, mu_prior=None, var_prior=None, dim=None, variational_beta=1.0):
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

    if mu_prior is None:
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    else:
        kl_divergence = kullback_leibler_divergence(dim, mu_prior, var_prior, mu, logvar)
    kld = torch.mean(kl_divergence)
    loss = torch.mean(args.lambda_vision * recon_loss_vision
                      + args.lambda_audio * recon_loss_audio
                      + args.lambda_tactile * recon_loss_tactile
                      + variational_beta * kld)
    return loss, recon_loss_vision, recon_loss_audio, recon_loss_tactile


def kullback_leibler_divergence(dim, cpu_mu_1, cpu_var_1, mu_2, logvar_2, device='mps'):
    mu_1 = torch.from_numpy(cpu_mu_1).to(torch.float32).to(device)
    var_1 = torch.from_numpy(cpu_var_1).to(torch.float32).to(device)
    logvar_1 = torch.log(var_1)
    var_2 = torch.exp(logvar_2)

    mu_diff = mu_1 - mu_2
    var_division = var_2 / var_1
    logvar_division = logvar_1 - logvar_2
    diff_division = mu_diff * mu_diff / var_1
    kld = 0.5 * ((var_division + logvar_division + diff_division).sum(1) - dim)
    return kld
