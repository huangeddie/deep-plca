import math

import torch
from torch import nn

from models import Encoder, Decoder


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