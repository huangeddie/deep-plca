import torch
from torch import nn
from torch.nn import functional as F


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