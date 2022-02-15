import torch
from torch import nn
import numpy as np
import sys
sys.path.append('..')
from TedSeg import dice_loss

def mint_ent_loss(x):
    '''
    TODO
    '''
    return 0

class SegLoss(nn.Module):
    def __init__(self, mode='supervised', mint_ent_w=0.005):
        super().__init__()
        self.mode = mode # supervised, unsupervised
        self.mint_ent_w = mint_ent_w
        self.mint_ent_loss = MinEntLoss()

    def forward(self, x, y, content):
        if self.mode == 'unsupervised':
            with torch.no_grad():
                bs = x.shape[0]
                indices = np.arange(bs)

                target_content_mask = content.detach().cpu().numpy().astype(bool)
                target_content_indices = torch.Tensor(indices[target_content_mask]).long().cuda()
                source_content_indices = torch.Tensor(indices[~target_content_mask]).long().cuda()

            x_seg = x.index_select(0, source_content_indices)
            y_seg = y.index_select(0, source_content_indices)
            dice = dice_loss(x_seg, y_seg)
            
            x_mint_ent = x.index_select(0, target_content_indices)
            mint_ent = mint_ent_loss(x_mint_ent)
            
            return dice + self.mint_ent_w*mint_ent
        return dice_loss(x, y)

class MinEntLoss(nn.Module):
    def __init__(self, ita=2., logits = True):
        super(MinEntLoss, self).__init__()
        self.ita = ita

    def forward(self, x):
        if self.logits:
            P = torch.cat([torch.sigmoid(x), 1 - torch.sigmoid(x)], axis = 1) + 1e-10
        else:
            P = x + 1e-10
        logP = torch.log(P)

 

        PlogP = P * logP
        ent = -1.0 * PlogP.sum(dim=1)
        ent = ent / 0.69
        ent = ent ** 2.0 + 1e-8
        ent = ent ** self.ita
        
        return self.weight* ent.mean()

def cofo_loss(mode='supervised', mint_ent_w=0.005, margin=0.3):
    return {
        'segmentation': SegLoss(mode=mode, mint_ent_w=mint_ent_w),
        'contrastive': nn.CosineEmbeddingLoss(margin=margin)
    }