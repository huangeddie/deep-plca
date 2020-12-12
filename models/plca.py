import torch
from torch import nn
from torch.nn import functional as F


class PLCA(nn.Module):
    pass


class DeepPLCA(PLCA):
    """
    Priors and impulse are deep CNN functions of the image, while the features are global parameters
    """

    def __init__(self, channels, imsize, nkern, kern_size, nconvs, hdim):
        super().__init__()
        self.nkern = nkern

        assert nconvs >= 1, nconvs

        # Priors and impulse

        # Initial convolutions
        impulse_size = imsize - kern_size + 1
        impulse = [
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, hdim, kern_size)
        ]

        prior = [
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, hdim, kern_size)
        ]

        # Add additional non-linear convolution layers
        for i in range(nconvs - 1):
            out_dim = nkern if i == nconvs - 2 else hdim
            impulse.extend([
                nn.BatchNorm2d(hdim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hdim, out_dim, 3, 1, 1),
            ])

            prior.extend([
                nn.BatchNorm2d(hdim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hdim, out_dim, 3, 1, 1),
            ])

        # Softmax impulse
        impulse.extend([
            nn.Flatten(start_dim=2),
            nn.Softmax(-1),
            nn.Unflatten(dim=2, unflattened_size=(impulse_size, impulse_size))
        ])

        # Pool and softmax prior
        prior.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Softmax(dim=1)
        ])

        # Make into modules
        self.impulse = nn.Sequential(*impulse)
        self.prior = nn.Sequential(*prior)

        # Features
        self.feat_logits = nn.Parameter(torch.randn(nkern, channels, kern_size, kern_size))

        self.softmax_feats = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Softmax(dim=-1),
            nn.Unflatten(1, (channels, kern_size, kern_size))
        )

    def forward(self, imgs):
        # Features
        feats = self.softmax_feats(self.feat_logits)

        # Impulse
        impulse = self.impulse(imgs)

        # Priors
        priors = self.prior(imgs)

        # Convolutional transpose
        recon = F.conv_transpose2d(priors * impulse, feats)

        # For some reason, when run on CUDA, there can be negative values
        recon.clamp_(min=0)

        return recon, priors, impulse, feats


def project_simplex_sort(v, z=1):
    """
    Projects a vector onto the simplex
    Takes O(d log d) time where d is the dimension of the vector.
    https://eng.ucmerced.edu/people/wwang5/papers/SimplexProj.pdf
    """
    n_features = v.shape[0]
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, 0) - z
    ind = torch.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = torch.maximum(v - theta, torch.zeros_like(v))
    return w


class ProjConvPLCA(PLCA):
    """
    Let params be the core nkern x channels x kern_size x kern_size parameters that influences everything
    The impulse convolutional kernels are generated from a learnable per-kernel affine transformation from params
    The feature logits are generated from a learnable per-kernel linear transformation from params
    (it’s linear and not affine because the feature logits are then immediately fed into the soft max activation which is shift invariant
    The priors are global
    """

    def __init__(self, channels, nkern, kern_size):
        super().__init__()
        self.nkern = nkern

        # Core parameters
        self.feats = nn.Parameter(torch.rand(nkern, channels, kern_size, kern_size))

        # Priors
        self.priors = nn.Parameter(torch.rand(1, nkern, 1, 1))

        # Project parameters to simplex
        self.project_params_to_simplex()

    def project_params_to_simplex(self):
        """
        This function must be called every time after the gradient descent step.
        """
        # Priors
        simplex_priors = self.priors.detach()
        priors_shape = self.priors.shape
        simplex_priors = project_simplex_sort(simplex_priors.flatten()).view(priors_shape)
        self.priors.data.copy_(simplex_priors)

        # Features
        simplex_feats = self.feats.detach()
        kern_shape = self.feats.shape[1:]
        for i in range(self.nkern):
            simplex_feats[i] = project_simplex_sort(simplex_feats[i].flatten()).view(kern_shape)
        self.feats.data.copy_(simplex_feats)

    def forward(self, imgs):
        # Impulse
        impulse_logits = F.conv2d(imgs, self.feats)
        impulse = impulse_logits / torch.sum(impulse_logits, dim=(2, 3), keepdim=True)

        # Convolutional transpose
        recon = F.conv_transpose2d(self.priors * impulse, self.feats)

        # For some reason, when run on CUDA, there can be negative values
        # recon.clamp_(min=0)

        return recon, self.priors.detach(), impulse, self.feats.detach()


class SoftConvPLCA(PLCA):
    """
    Let params be the core nkern x channels x kern_size x kern_size parameters that influences everything
    The impulse convolutional kernels are generated from a learnable per-kernel affine transformation from params
    The feature logits are generated from a learnable per-kernel linear transformation from params
    (it’s linear and not affine because the feature logits are then immediately fed into the soft max activation which is shift invariant
    The priors are global
    """

    def __init__(self, channels, imsize, nkern, kern_size):
        super().__init__()
        self.nkern = nkern

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
