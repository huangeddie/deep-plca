import os

import torch

from models.auto import AutoLayer, AutoEncoder
from models import plca


def make_model(args, channels):
    if args.model == 'proj-conv-plca':
        model = plca.ProjConvPLCA(channels, args.nkern, args.kern_size)
    elif args.model == 'soft-conv-plca':
        model = plca.SoftConvPLCA(channels, args.imsize, args.nkern, args.kern_size)
    elif args.model == 'deep-plca':
        model = plca.DeepPLCA(channels, args.imsize, args.nkern, args.kern_size, args.nconvs, args.hdim)
    elif args.model == 'ae':
        model = AutoEncoder(channels, args.imsize, args.zdim)
    elif args.model == 'al':
        model = AutoLayer(channels, args.imsize, args.zdim)
    else:
        raise Exception(f'unknown model {args.model}')
    return model


def optionally_load_wts(args, model):
    if os.path.exists(args.save) and args.load:
        model.load_state_dict(torch.load(args.save))
        print('loaded saved weights', flush=True)
    else:
        print('starting with new weights', flush=True)
