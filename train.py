import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from tqdm.auto import tqdm

import models


class Metrics:
    def __init__(self):
        self._loss = {'train': [], 'test': []}

    def add_epoch_loss(self, subset, loss):
        self._loss[subset].append(loss)

    def loss(self, subset):
        return np.array(self._loss[subset]).flatten()

    def nepochs(self):
        n = len(self._loss['train'])
        return n

    def status_str(self):
        m = [np.mean(self._loss['train'][-1]),
             np.mean(self._loss['test'][-1])]
        return f'(train/test): {m[0]:.3}/{m[1]:.3} loss'


#### Losses

def entropy(X, dim):
    # Assumes that values along the given dimension(s) are probabilities that sum to 1
    p_sums = torch.sum(X, dim=dim)
    assert torch.allclose(p_sums, torch.ones_like(p_sums)), p_sums
    entropy = -torch.sum(X * torch.log(X + 1e-7), dim=dim)
    return entropy.mean()


def l2_loss(X, dim):
    # Assumes that values along the given dimension(s) are probabilities that sum to 1
    p_sums = torch.sum(X, dim=dim)
    assert torch.allclose(p_sums, torch.ones_like(p_sums)), p_sums
    return -torch.sum(X ** 2, dim=dim).mean()


def cross_entropy(inputs, targets, dim):
    ce = -torch.sum(targets * torch.log(inputs + 1e-7), dim=dim)
    return ce.mean()


def kl_div(inputs, targets, dim):
    ce = -torch.sum(targets * torch.log(inputs + 1e-7), dim=dim)
    H = entropy(targets, dim)
    return (ce - H).mean()


def get_recon_loss(args, recon, imgs):
    assert (recon >= 0).all() and (recon <= 1).all()
    if args.recon == 'ce':
        recon_loss = cross_entropy(recon, imgs, dim=(1, 2, 3))
    elif args.recon == 'kl':
        recon_loss = kl_div(recon, imgs, dim=(1, 2, 3))
    elif args.recon == 'bce':
        recon_loss = F.binary_cross_entropy(recon, imgs)
    elif args.recon == 'mse':
        recon_loss = F.mse_loss(recon, imgs)
    else:
        raise Exception(f'unknown reconstruction loss {args.recon}')
    return recon_loss


#### Steps

def pcla_step(args, model, imgs):
    # Deep PCLA
    recon, priors, impulse, feat = model(imgs)

    # Entropy loss
    prior_entropy = entropy(priors, dim=[1, 2, 3])
    impulse_entropy = entropy(impulse, dim=[2, 3])
    feat_entropy = entropy(feat, dim=[1, 2, 3])

    # Reconstruction loss
    recon_loss = get_recon_loss(args, recon, imgs)

    # Optimize over total loss
    loss = recon_loss + args.beta1 * prior_entropy + args.beta2 * impulse_entropy + args.beta3 * feat_entropy
    return loss, recon_loss


def ae_step(args, model, imgs):
    # AutoEncoder
    recon = model(imgs)

    # Reconstruction loss
    recon_loss = get_recon_loss(args, recon, imgs)

    return recon_loss, recon_loss


def al_step(args, model, imgs):
    _, all_layer_logits = model(imgs)

    # Average reconstruction loss over all layers
    loss = 0
    recon_logits = torch.zeros_like(imgs)
    for logits in all_layer_logits:
        recon_logits += logits
        loss += get_recon_loss(args, torch.sigmoid(recon_logits), imgs)

    return loss / len(all_layer_logits), None


#### Loops

def loop_data(args, model, data_loader, opt=None):
    training = opt is not None
    torch.set_grad_enabled(training), model.cuda()

    losses = []
    pbar = tqdm(data_loader, 'train' if training else 'test', mininterval=1,
                leave=False, position=1)
    model.train(training)
    for imgs, _ in pbar:
        # Setup
        imgs = imgs.cuda()
        if training:
            opt.zero_grad()

            # Train steps defined seperately for each model
        if isinstance(model, models.ConvPCLA):
            loss, recon_loss = pcla_step(args, model, imgs)
        elif isinstance(model, models.AutoEncoder):
            loss, recon_loss = ae_step(args, model, imgs)
        elif isinstance(model, models.AutoLayer):
            loss, recon_loss = al_step(args, model, imgs)
        else:
            raise Exception(f'unknown model {model}')

        # Backprop?
        if training:
            loss.backward()
            opt.step()

        # Record
        losses.append(recon_loss.item())
        pbar_str = f'{losses[-1]:.3} loss'
        pbar.set_postfix_str(pbar_str, refresh=False)

    return losses


def train(args, model, train_loader, test_loader):
    # Setup
    losses = []
    metrics = Metrics()

    # Optimizer
    if args.opt == 'adam':
        opt = optim.Adam(model.parameters(), args.lr)
    elif args.opt == 'sgd':
        opt = optim.SGD(model.parameters(), args.lr, momentum=0.9)
    else:
        raise Exception(f'Unknown optimizer {args.opt}')

    # Train
    pbar = tqdm(range(args.epochs))
    for i in pbar:
        try:
            # Train
            loss = loop_data(args, model, train_loader, opt)
            metrics.add_epoch_loss('train', loss)

            # Save
            torch.save(model.state_dict(), './model.pt')

            # Test
            loss = loop_data(args, model, test_loader)
            metrics.add_epoch_loss('test', loss)

            pbar.set_postfix_str(metrics.status_str())
        except KeyboardInterrupt:
            break

    return metrics
