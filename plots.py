import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

import models


def plot_metrics(metrics):
    # Plot training metrics
    f, ax = plt.subplots(1, 1)
    f.set_size_inches(15, 4)

    nepochs = metrics.nepochs()

    # Loss
    ax.set_title('loss')
    ax.set_xlabel('epochs')
    for subset in ['train', 'test']:
        loss = metrics.loss(subset)
        ax.plot(np.linspace(0, nepochs, len(loss)), loss, label=subset)
    ax.legend()


def plot_plca_recon(args, sample_img, model):
    torch.set_grad_enabled(False)
    model.eval(), model.cuda()

    # Sample reconstruction
    recon, priors, impulses, feats = model(sample_img.cuda())

    # Assert reconstructions are probability distributions
    recon_sum = recon.sum(dim=(1, 2, 3))
    assert torch.allclose(recon_sum, torch.ones_like(recon_sum)), recon_sum

    # Plot
    f, ax = plt.subplots(2, 3)
    f.set_size_inches(30, 20)

    # Priors
    nrow = max(int(model.nkern ** 0.5), 1)
    ax[0, 0].set_title('priors')
    grid_priors = make_grid(priors[0].unsqueeze(1).cpu(), nrow=nrow)
    pcm = ax[0, 0].imshow(grid_priors[0])
    f.colorbar(pcm, ax=ax[0, 0])

    # Impulses
    grid_impulses = make_grid(impulses[0].unsqueeze(1).cpu(), nrow=nrow, pad_value=1)
    ax[0, 1].set_title('impulses')
    pcm = ax[0, 1].imshow(grid_impulses[0])
    f.colorbar(pcm, ax=ax[0, 1])

    # Features
    # Normalize each feature individually
    max_feats, _ = feats.flatten(1).max(dim=1, keepdim=True)
    feats = feats / max_feats.unflatten(1, (1, 1, 1))
    grid_feats = make_grid(feats.cpu(), nrow=nrow, normalize=True)
    ax[0, 2].set_title('normalized features')
    ax[0, 2].imshow(grid_feats.permute(1, 2, 0))

    # Original
    ax[1, 0].set_title('normalized image')
    img = make_grid(sample_img.cpu(), normalize=True)
    ax[1, 0].imshow(img.permute(1, 2, 0))

    # Reconstruction
    ax[1, 1].set_title('normalized reconstruction')
    img = make_grid(recon.cpu(), normalize=True)
    ax[1, 1].imshow(img.permute(1, 2, 0))

    ax[1, 2].remove()

    f.savefig(os.path.join(args.outdir, 'recon.jpg'))

    # Plot top components
    f, ax = plt.subplots(1, 6)
    f.set_size_inches(24, 4)
    f.suptitle('top 6 components')
    # Sort by priors
    top_kerns = torch.argsort(priors, dim=1, descending=True)
    for i in range(6):
        idx = top_kerns[0, i]
        component = F.conv_transpose2d(impulses[:, idx:idx + 1], feats[idx:idx + 1])
        component.clamp_(min=0)
        img = make_grid(component.cpu(), normalize=True)
        ax[i].imshow(img.permute(1, 2, 0))

    f.savefig(os.path.join(args.outdir, 'comp.jpg'))


def plot_ae_recon(args, sample_img, model):
    torch.set_grad_enabled(False)
    model.eval()

    # Sample reconstruction
    recon = model(sample_img.cuda())

    # Plot
    f, ax = plt.subplots(1, 2)
    f.set_size_inches(12, 5)

    # Original
    ax[0].set_title('normalized image')
    img = make_grid(sample_img.cpu(), normalize=True)
    ax[0].imshow(img.permute(1, 2, 0))

    # Reconstruction
    ax[1].set_title('normalized reconstruction')
    img = make_grid(recon.cpu(), normalize=True)
    ax[1].imshow(img.permute(1, 2, 0))


def plot_al_recon(args, sample_img, model):
    torch.set_grad_enabled(False)
    model.cuda(), model.eval()

    # Sample reconstruction
    recon, layer_logits = model(sample_img.cuda())
    n = len(layer_logits)

    # Plot
    f, ax = plt.subplots(2, 1 + n)
    ax[1, 0].remove()
    f.set_size_inches(4 * n, 8)

    # Original
    ax[0, 0].set_title('image')
    img = make_grid(sample_img.cpu())
    ax[0, 0].imshow(img.permute(1, 2, 0))

    # Layers
    recon_layer_logits = torch.zeros_like(recon)
    for i in range(n):
        ax[0, i + 1].set_title(f'layer {i + 1}')
        img = make_grid(torch.sigmoid(layer_logits[i]).cpu())
        ax[0, i + 1].imshow(img.permute(1, 2, 0))

        ax[1, i + 1].set_title(f'layer {i + 1} reconstruction')
        recon_layer_logits += layer_logits[i]
        img = make_grid(torch.sigmoid(recon_layer_logits).cpu())
        ax[1, i + 1].imshow(img.permute(1, 2, 0))


def plot_recon(args, imgs, model):
    if isinstance(model, models.plca.ConvPLCA):
        plot_plca_recon(args, imgs, model)
    elif isinstance(model, models.auto.AutoEncoder):
        plot_ae_recon(args, imgs, model)
    elif isinstance(model, models.auto.AutoLayer):
        plot_al_recon(args, imgs, model)
    else:
        raise Exception(f'unknown model {model}')
