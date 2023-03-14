import torch.utils.data as data
import numpy as np
import os
from PIL import Image
import random
import cv2

class ColorDataset(data.Dataset):
    def __init__(self, conf):
        super(ColorDataset, self).__init__()
        # import ipdb; ipdb.set_trace()
        data_root, split, fake = conf.data_dir, conf.split, conf.fake
        self.data_root = data_root
        self.split = split
        self.fake = fake
        self.image_paths = self.glob_imgs(os.path.join(self.data_root, self.split))
        self.image_fake_paths = self.glob_imgs(os.path.join(self.data_root, self.fake))
        

    def __getitem__(self, index):
        img_size = (256,256)
        image_path = self.image_paths[index]
        img_RGB = np.array(Image.open(image_path).convert('RGB').resize(img_size)).astype(np.float32) / 255.0
        img_GREY = np.array(Image.open(image_path).convert('L').resize(img_size)).astype(np.float32)
        # import ipdb; ipdb.set_trace()
        image_fake_path = self.image_fake_paths[random.randint(0, len(self.image_fake_paths)-1)]
        img_FAKE = np.array(Image.open(image_fake_path).convert('RGB').resize(img_size)).astype(np.float32) / 255.0
        # img_RGB = cv2.resize(img_RGB, img_size, interpolation=cv2.INTER_CUBIC)
        # img_GREY = cv2.resize(img_GREY, img_size, interpolation=cv2.INTER_CUBIC)
        # img_FAKE = cv2.resize(img_FAKE, img_size, interpolation=cv2.INTER_CUBIC)
        ret = {
            'img_RGB': img_RGB,
            'img_GREY': img_GREY,
            'img_FAKE': img_FAKE
        }
        
        return ret

    def __len__(self):
        return len(self.image_paths)
    
    def glob_imgs(self, path):
        from glob import glob
        imgs = []
        for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
            imgs.extend(glob(os.path.join(path, ext)))
        return imgs