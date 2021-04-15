import os
import torch.utils.data as data
from PIL import Image
from torchvision.datasets.folder import is_image_file, default_loader
from combinet.src.dataset.utils import LabelToLongTensor
import torch 
import numpy as np

def _make_dataset(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                item = path
                images.append(item)
    return images


CLASS_WEIGHT_BACTERIA = torch.FloatTensor([3.,  1.,  0.1])
mean = [0.16823019, 0.20769378, 0.26853953]
std = [0.22752789, 0.25381214, 0.25989399]

class Bacteria(data.Dataset):
    colors = [
        (128, 128, 128),
        (0, 128, 192),
        (0, 0, 0),
    ]
    ignore_index = 2

    def __init__(self, root, split='train', joint_transform=None,
                 transform=None, target_transform=LabelToLongTensor()):
        self.root = root+"bacteria"
        assert split in ('train', 'val', 'test')
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.loader = default_loader

        self.imgs = _make_dataset(os.path.join(self.root, self.split))

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        target = Image.open(path.replace(self.split, self.split + 'annot'))
        
        if self.joint_transform is not None:
            img, target = self.joint_transform([img, target])

        if self.transform is not None:
            img = self.transform(img)
        target = self.target_transform(np.array(target))
        return img, target

    def __len__(self):
        return len(self.imgs)

