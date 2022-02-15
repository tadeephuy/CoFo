
import torch
import numpy as np
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
import sys
sys.path.append('..')
from TedSeg import SegmentationDataset


class CoFoSegmentationDataset(SegmentationDataset):
    def __init__(self, source_df, target_df, transforms, size):
        self.s_df = source_df
        self.t_df = target_df
        self.transforms = transforms(size)
    
    def __getitem__(self, idx):
        xs, ys = self.s_df.iloc[idx][['image', 'mask']]
        xt, yt = self.t_df.sample().iloc[0][['image', 'mask']]
        
        xs, xt = map(self.imread, [xs, xt])
        ys, yt = map(self.mskread, [ys, yt])
        
        sample_s = self.transforms(image=xs, mask=ys)
        xs, ys = sample_s['image'], sample_s['mask']
        
        sample_t = self.transforms(image=xt, mask=yt)
        xt, yt = sample_t['image'], sample_t['mask']
        
        xs, xt = map(self.transpose, [xs, xt])
        xs, xt = map(torch.tensor, [xs, xt])
        ys, yt = map(ToTensor(), [ys, yt])
        
        return xs, ys, xt, yt
    
    def __len__(self): return len(self.s_df)
    
    def show_img(self, beta=0.09):
        idx = np.random.randint(len(self))
        xs, ys, xt, yt = self[idx]

        blend_s = self.create_blend(xs/255, ys, denormalize=False)
        blend_t = self.create_blend(xt/255, yt, denormalize=False)
        
        blend = np.concatenate([blend_s, blend_t], axis=1)      

        plt.figure(figsize=(10,20))
        plt.imshow(blend)

    @staticmethod
    def transpose(x): return np.transpose(x, axes=(2,0,1))