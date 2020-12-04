import os

import torch

from models.ae import AutoLayer, AutoEncoder
from models.deep_plca import ConvPCLA


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
