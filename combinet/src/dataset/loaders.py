import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from combinet.src.dataset.camvid import CamVid
from combinet.src.dataset.camvid import mean as camvid_mean
from combinet.src.dataset.camvid import std as camvid_std
from combinet.src.dataset.bacteria import Bacteria
from combinet.src.dataset.bacteria import mean as bacteria_mean
from combinet.src.dataset.bacteria import std as bacteria_std
from combinet.src.dataset import joint_transforms
import logging
from PIL import Image

def get_train_loaders(args):
    train_dset = None
    valid_dset = None
    if args.dataset=="camvid":
        train_joint_transformer = transforms.Compose([
            joint_transforms.JointRandomSizedCrop(args.crop_size, ignore_index=CamVid.ignore_index),
            joint_transforms.JointRandomHorizontalFlip()
        ])
        normalize = transforms.Normalize(mean=camvid_mean, std=camvid_std, inplace=True)
        train_dset = CamVid(args.data, 'train',
                                joint_transform=train_joint_transformer,
                                transform=transforms.Compose([
                                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
                                    transforms.ToTensor(),
                                    normalize
                                    ]))
        valid_dset = CamVid(
            args.data, 'val', joint_transform=None,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize
            ]))
    elif args.dataset=="bacteria":
        train_joint_transformer = transforms.Compose([
            joint_transforms.JointRandomSizedCrop(args.crop_size, ignore_index=Bacteria.ignore_index),
            joint_transforms.JointRandomHorizontalFlip(),
            joint_transforms.JointRandomVerticalFlip()
        ])
        normalize = transforms.Normalize(mean=bacteria_mean, std=bacteria_std, inplace=True)
        train_dset = Bacteria(args.data, 'train',
                                joint_transform=train_joint_transformer,
                                transform=transforms.Compose([
                                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
                                    transforms.ToTensor(),
                                    normalize
                                    ]))
        valid_dset = Bacteria(
            args.data, 'val', joint_transform=None,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize
            ]))
    else:
        raise NotImplementedError("Other datasets not implemented")
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.train_batch_size,
        shuffle=True,
        pin_memory=args.gpu != -1,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dset,
        batch_size=args.valid_batch_size,
        shuffle=False,
        pin_memory=args.gpu != -1,
    )


    logging.info('### Train size: {}, Validation size: {} ###'.format(
            len(train_loader.dataset), len(valid_loader.dataset)))

    return train_loader, valid_loader


def get_test_loader(args, shuffle=False):
    test_dset = None
    if args.dataset == "camvid":
        normalize = transforms.Normalize(mean=camvid_mean, std=camvid_std, inplace=True)
        test_joint_transform = None 
        test_dset = CamVid(
            args.data, 'test', joint_transform=test_joint_transform,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize
            ]))
    elif args.dataset == "bacteria":
        normalize = transforms.Normalize(mean=bacteria_mean, std=bacteria_std, inplace=True)
        test_joint_transformer = None 
        test_dset = Bacteria(
            args.data, 'test', joint_transform=test_joint_transformer,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize
            ]))
    elif args.dataset == "random":
        test_dset = datasets.VOCSegmentation(args.data, '2012', 'val', True, 
        transform=transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BILINEAR),
            transforms.ToTensor()
        ]),
        target_transform=transforms.Compose([
                transforms.Resize((224, 224), interpolation=Image.NEAREST),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.squeeze().long()),
            ]))
        test_dset = torch.utils.data.Subset(test_dset, range(250))
    else:
        raise NotImplementedError("Other datasets not implemented")
    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=args.test_batch_size,
        shuffle=shuffle,
        pin_memory=args.gpu != -1,
    )

    logging.info('### Test size: {} ###'.format(len(test_loader.dataset)))
    return test_loader
    
