from __future__ import division
import math
import random
from PIL import Image
import torchvision.transforms.functional as F


class JointRandomHorizontalFlip(object):
    def __call__(self, imgs):
        if random.random() < 0.5:
            return F.hflip(imgs[0]), F.hflip(imgs[1])
        return imgs[0], imgs[1]

class JointRandomVerticalFlip(object):
    def __call__(self, imgs):
        if random.random() < 0.5:
            return F.vflip(imgs[0]), F.vflip(imgs[1])
        return imgs[0], imgs[1]


class JointRandomSizedCrop(object):
    def __init__(self, size=360, bounds = (0.5, 2.0), ignore_index=11):
        self.size = size
        self.bounds = bounds
        self.ignore_index = ignore_index

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th = tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, imgs):
        img, lbl = imgs
        assert img.size == lbl.size
        h = img.size[0]
        w = img.size[1]
        area = img.size[0] * img.size[1]

        scale = random.uniform(self.bounds[0], self.bounds[1])
        target_area = scale * area
        aspect_ratio = random.uniform(3. / 4, 4. / 3)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        target_size = (w, h)
        img = F.resize(img, target_size, Image.BILINEAR) 
        lbl = F.resize(lbl, target_size, Image.NEAREST)

        if img.size[0] < self.size:
            img = F.pad(img, padding=int((1 + self.size - img.size[0]) / 2))
            lbl = F.pad(lbl,fill=self.ignore_index, padding=int((1 + self.size - lbl.size[0]) / 2))

        if img.size[1] < self.size:
            img = F.pad(img, padding=int((1 + self.size - img.size[1]) / 2))
            lbl = F.pad(lbl, fill=self.ignore_index, padding=int((1 + self.size - lbl.size[1]) / 2))

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(lbl, i, j, h, w)

