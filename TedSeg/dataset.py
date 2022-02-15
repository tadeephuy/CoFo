from torch.utils.data import Dataset
import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import albumentations as albu
from albumentations.pytorch import transforms as torch_albu
from matplotlib import pyplot as plt
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, df, transforms, size, normalize=True):
        self.df = df
        self.images = self.df['image'].values
        self.masks = self.df['mask'].values
        self.transforms = transforms(size)
        self.normalize = normalize # convenient function to convert range 0 1 + normalize

    def __getitem__(self, idx):
        image, mask = self.images[idx], self.masks[idx]
        
        image = self.imread(image)
        mask = self.mskread(mask)
        
        sample = self.transforms(image=image, mask=mask)

        image, mask = sample['image'], sample['mask']
        
        image = torch.tensor(np.transpose(image, (2,0,1)))/255
        mask = ToTensor()(mask)

        if self.normalize:
            image = (image - 0.5)/0.22
            
        return image, mask
    
    def __len__(self): return len(self.df)
    
    def show_img(self, idx=None):
        if idx is None:
            idx = np.random.randint(len(self))
        img, msk = self[idx]
        blend = self.create_blend(img, msk, denormalize=self.normalize)
        plt.figure(figsize=(10,10))
        plt.imshow(blend)
    
    def imread(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def mskread(self, path):
        mask = cv2.imread(path, 0).astype(np.uint8)
        return mask
    
    @staticmethod
    def create_blend(img, msk, denormalize=True):
        if denormalize:
            img = (img * 0.22) + 0.5
        img = img.permute(1,2,0).detach().numpy()
        msk = msk.permute(1,2,0)[...,0].detach().numpy()
        mask = msk[...,None]
        color_mask = np.array([0.13*msk, 0.15*msk, 0.9*msk])
        color_mask = np.transpose(color_mask, (1,2,0))
        blend = 0.35*color_mask + 0.65*img*mask + (1 - mask)*img
        return blend

def get_training_augmentation(img_size):
    train_transform = [
#         albu.Resize(img_size, img_size, always_apply=True),
        albu.OneOf([
#             albu.RandomGamma(gamma_limit=(60, 120), p=0.9),
            albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
#             albu.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.9),
        ]),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(0.1, 0.2, rotate_limit=90, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
        albu.SmallestMaxSize(img_size),
        albu.CenterCrop(img_size, img_size),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation(img_size):
    test_transform = [

#         albu.Resize(img_size, img_size, always_apply=True),
#         albu.CLAHE(clip_limit=(2., 2.), tile_grid_size=(4, 4), always_apply=True),
        albu.SmallestMaxSize(img_size),
        albu.CenterCrop(img_size, img_size),
    ]
    return albu.Compose(test_transform)