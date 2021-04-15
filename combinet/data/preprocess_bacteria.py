import numpy as np
import os
import fnmatch
from shutil import copyfile
import random 
from PIL import Image
import collections
from tqdm import tqdm

def create_datasets(root):
    INPUT_DIR = root+"/bacteria/"
    files = fnmatch.filter(os.listdir(INPUT_DIR+'/images'), '*.png')

    # Infer the class distribution 
    hist = None
    count = 0
    for file in tqdm(files):
        img = Image.open(root+"/bacteria/masks/"+file)
        img = np.array(img)[:,:,0]
        d = collections.Counter(img.reshape(-1))
        if hist is None:
            hist = d
        else:
            hist = dict(collections.Counter(hist)+d)
        count+=np.prod(img.shape)
    print(hist)
    print(count)
    pixel_counts = np.array(list(hist.values()), dtype=np.float64)[::-1]
    pixel_counts/=np.array(count) 
    print((1/8)/pixel_counts)
    #should read: [11.70825808  0.95896916  0.1455222 ]

    # Infer the normalisation constants
    all_images = []
    crop_size = 280
    for file in tqdm(files):
        img = Image.open(root+"/bacteria/images/"+file)
        width, height = img.size   # Get dimensions
        left = (width - crop_size)/2
        top = (height - crop_size)/2
        right = (width + crop_size)/2
        bottom = (height + crop_size)/2
        # Crop the center of the image
        img = img.crop((left, top, right, bottom))
        all_images.append(np.array(img, dtype=np.float64)/255)
    all_images = np.stack(all_images, axis=0)
    print(all_images.shape)
    print(np.mean(all_images, axis=(0,1,2)))
    print(np.std(all_images, axis=(0,1,2)))
    #should read: [0.16823019 0.20769378 0.26853953], [0.22752789 0.25381214 0.25989399]

    random.shuffle(files)
    train, validate, test = np.split(files, [int(0.6*len(files)), int((0.6+ 0.2)*len(files))])
    def copy_files(files, split, root):
        try:
            os.mkdir(root+"/bacteria/"+split+"/")
            os.mkdir(root+"/bacteria/"+split+"annot/")
        except:
            pass
        for file in files:
            copyfile(root+"/bacteria/"+"images/"+file, root+"/bacteria/"+split+"/"+file)
            mask = np.array(Image.open(root+"/bacteria/"+"masks/"+file), dtype=np.int8)[:,:,0]
            new_mask = np.zeros_like(mask)
            new_mask[mask==0] = 2
            new_mask[mask==1] = 1 
            new_mask[mask==2] = 0
            new_mask = Image.fromarray(new_mask)
            new_mask.save(root+"/bacteria/"+split+"annot"+"/"+file)

    copy_files(train, "train", root)
    copy_files(validate, "val", root)
    copy_files(test, "test", root)

np.random.seed(1)
random.seed(1)
create_datasets(".")



