import torch
from torch import nn
import numpy as np
import sys
sys.path.append('..')
from models.unet import Unet, PooledSelfAttention2d
from .gradrev import GradientReversal

class CoFoUnet(Unet):
    def __init__(self, emb_size=512, **kwargs):
        super().__init__(**kwargs)
        self.grad_rev = GradientReversal(alpha=1.)
        self.style_head = nn.Sequential(
            PooledSelfAttention2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.InstanceNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.InstanceNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 1, 1), nn.InstanceNorm2d(1), nn.LeakyReLU(),
            nn.AvgPool2d(2), nn.Flatten(), nn.Linear(3136, emb_size)
        )

        self.content_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 1, 1), nn.BatchNorm2d(1), nn.LeakyReLU(),
            nn.AvgPool2d(2), nn.Flatten(), nn.Linear(3136, emb_size)
        )

    def forward(self, x, mode='segmentation'):
        if mode=='contrastive':
            # only feed through content and style head
            x = self.grad_rev(x)
            style = self.style_head(x)
#             content = self.content_head(x.detach()) # block gradient flow from content head to encoder
            content = self.content_head(x)
            return style, content

        x, features_value = self.forward_backbone(x)
        for i, block in enumerate(self.blocks):
            name = self.features_name[i]
            x = block(x, features_value[name])
        
        if mode=='segmentation':
            # default get mask
            mask = self.out_conv(x)
            return mask
        if mode=='feature':
            # get mask and intermediate featre
            mask = self.out_conv(x)
            return mask, x            