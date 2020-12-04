import os

import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    """
    Encodes an image of any size into a zdim x 1 x 1 embedding
    """

    def __init__(self, channels, zdim):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, zdim, 3, padding=1),
            nn.BatchNorm2d(zdim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, imgs):
        return self.encoder(imgs)


class Decoder(nn.Module):
    """
    Decodes a zdim x 1 x 1 embedding into an
    channels x imsize x imsize image
    """

    def __init__(self, zdim, channels, imsize, hdim):
        super().__init__()
        upsamples = int(math.log2(imsize))
        assert upsamples > 0

        # Encoder
        decoder = []
        for i in range(upsamples):
            if i == 0:
                decoder.extend([
                    nn.ConvTranspose2d(zdim, hdim, 3, 2, 1, 1),
                    nn.BatchNorm2d(hdim),
                    nn.ReLU()
                ])
            else:
                decoder.extend([
                    nn.ConvTranspose2d(hdim, hdim, 3, 2, 1, 1),
                    nn.BatchNorm2d(hdim),
                    nn.ReLU()
                ])

        decoder.extend([
            nn.Upsample((imsize, imsize), mode='bicubic'),
            nn.ConvTranspose2d(hdim, channels, 3, 1, 1)
        ])
        self.decoder = nn.Sequential(*decoder)

    def forward(self, imgs):
        return self.decoder(imgs)


import math


class DrawLayer(nn.Module):
    def __init__(self, in_c, out_c, img_c, imsize):
        super().__init__()
        self.img_c = img_c

        self.conv = nn.ConvTranspose2d(in_c, out_c, 3, 2, 1, 1)

        self.act = nn.Sequential(
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        self.upsample = nn.Upsample((imsize, imsize), mode='bicubic')

    def forward(self, inp):
        x, layers = inp
        y = self.conv(x)

        # First img_c channels are also treated as the layers
        layer_logits = y[:, :self.img_c]
        upsampled_logits = self.upsample(layer_logits)
        layers.append(upsampled_logits)

        out = self.act(y)
        return (out, layers)


#### Auto layer

class AutoLayer(nn.Module):
    def __init__(self, channels, imsize, zdim):
        super().__init__()
        self.imsize = imsize

        # Image size should be a power of 2
        upsamples = math.log(imsize, 2)
        assert upsamples.is_integer()
        upsamples = int(upsamples)

        # Encoder
        self.encoder = Encoder(channels, zdim)

        draw_layers = []
        hdim = 128
        for i in range(upsamples):
            if i == 0:
                draw_layers.append(DrawLayer(zdim, hdim, channels, imsize))
            elif i == upsamples - 1:
                draw_layers.append(DrawLayer(hdim, channels, channels, imsize))
            else:
                draw_layers.append(DrawLayer(hdim, hdim, channels, imsize))

        self.draw_layers = nn.Sequential(*draw_layers)

    def forward(self, imgs):
        # Encode
        z = self.encoder(imgs)

        # Draw layers
        _, layers = self.draw_layers((z, []))
        recon_logits = torch.zeros_like(imgs)
        for layer in layers:
            recon_logits += layer
        return torch.sigmoid(recon_logits), layers


#### Auto encoder

class AutoEncoder(nn.Module):
    def __init__(self, channels, imsize, zdim):
        super().__init__()

        # Encoder
        self.encoder = Encoder(channels, zdim)

        # Decoder
        self.decoder = Decoder(zdim, channels, imsize, hdim=256)

    def forward(self, imgs):
        # Encode
        z = self.encoder(imgs)

        # Decode
        recon_logits = self.decoder(z)

        return torch.sigmoid(recon_logits)


#### Conv PCLA

class ConvPCLA(nn.Module):
    def __init__(self, channels, imsize, nkern, kern_size):
        super().__init__()
        self.nkern = nkern
        hdim = 256

        # Core parameters
        self.params = nn.Parameter(torch.randn(nkern, channels, kern_size, kern_size))

        # Norm image
        self.norm_img = nn.BatchNorm2d(channels)

        # Priors
        self.prior_logits = nn.Parameter(torch.randn(1, nkern, 1, 1))

        # Impulse
        self.impulse_w = nn.Parameter(torch.rand(nkern, 1, 1, 1))
        self.impulse_b = nn.Parameter(torch.rand(nkern, 1, 1, 1))

        impulse_size = imsize - kern_size + 1
        self.softmax_impulse = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Softmax(-1),
            nn.Unflatten(dim=2, unflattened_size=(impulse_size, impulse_size))
        )

        # Features a.k.a kernels
        self.feat_w = nn.Parameter(torch.rand(nkern, 1, 1, 1))

        self.softmax_feats = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Softmax(dim=-1),
            nn.Unflatten(1, (channels, kern_size, kern_size))
        )

    def impulse_affine(self, params):
        return self.impulse_w * params + self.impulse_b

    def feat_linear(self, params):
        return self.feat_w * params

    def forward(self, imgs):
        norm_imgs = self.norm_img(imgs)

        # Priors
        priors = F.softmax(self.prior_logits, dim=1)

        # Impulse
        impulse_kernels = self.impulse_affine(self.params)
        impulse_logits = F.conv2d(norm_imgs, impulse_kernels)
        impulse = self.softmax_impulse(impulse_logits)

        # Features
        feat_logits = self.feat_linear(self.params)
        feats = self.softmax_feats(feat_logits)

        # Convolutional transpose
        recon = F.conv_transpose2d(priors * impulse, feats)

        # For some reason, when run on CUDA, there can be negative values
        recon.clamp_(min=0)

        return recon, priors, impulse, feats


#### Model utilities

def make_model(args, channels):
    if args.model == 'pcla':
        model = ConvPCLA(channels, args.imsize, args.nkern, args.kern_size)
    elif args.model == 'ae':
        model = AutoEncoder(channels, args.imsize, args.zdim)
    elif args.model == 'al':
        model = AutoLayer(channels, args.imsize, args.zdim)
    else:
        raise Exception(f'unknown model {args.model}')
    return model


def optionally_load_wts(args, model):
    if os.path.exists('./model.pt') and not args.new:
        model.load_state_dict(torch.load('./model.pt'))
        print('loaded weights', flush=True)
    else:
        print('starting with new weights', flush=True)
