from torch.utils import data
from torchvision import transforms, datasets


def center_crop_square(im, size):
    width, height = im.size  # Get dimensions

    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im


def load_data(args, shuffle, droplast):
    # Transform to grayscale
    grayscale = transforms.Grayscale()
    transform = transforms.Compose([transforms.Lambda(lambda img: center_crop_square(img, min(*img.size))),
                                    transforms.Resize((args.imsize, args.imsize)),
                                    transforms.ToTensor(),
                                    grayscale])

    # Make probability distribution?
    if args.prob:
        transform.transforms.append(transforms.Lambda(lambda x: x / x.sum()))

    # Dataset
    channels = 1
    if args.data == 'mnist':
        # MNIST is already grayscale
        transform.transforms.remove(grayscale)
        train_data = datasets.MNIST('./', train=True, transform=transform, download=True)
        test_data = datasets.MNIST('./', train=False, transform=transform, download=True)
    elif args.data == 'kmnist':
        # KMNIST is already grayscale
        transform.transforms.remove(grayscale)
        train_data = datasets.KMNIST('./', train=True, transform=transform, download=True)
        test_data = datasets.KMNIST('./', train=False, transform=transform, download=True)
    elif args.data == 'cifar10':
        train_data = datasets.CIFAR10('./', train=True, transform=transform, download=True)
        test_data = datasets.CIFAR10('./', train=False, transform=transform, download=True)
    elif args.data == 'cifar100':
        train_data = datasets.CIFAR100('./', train=True, transform=transform, download=True)
        test_data = datasets.CIFAR100('./', train=False, transform=transform, download=True)
    else:
        raise Exception(f'unknown data {args.data}')

    # Dataloader
    train_loader = data.DataLoader(train_data, batch_size=args.bsz,
                                   shuffle=shuffle, num_workers=4,
                                   pin_memory=True, drop_last=droplast)
    test_loader = data.DataLoader(test_data, batch_size=args.bsz,
                                  shuffle=shuffle, num_workers=4,
                                  pin_memory=True, drop_last=droplast)

    return train_loader, test_loader, channels
