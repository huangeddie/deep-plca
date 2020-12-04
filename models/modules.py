import math

from torch import nn


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