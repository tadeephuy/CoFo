import torch
from torch import nn

def dice_loss(input, target):
    smooth = 1e-6
    
    iflat = torch.sigmoid(input).view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

def batch_dice_score(input, target):
    smooth = 1e-6
    
    bs = input.shape[0]
    iflat = torch.sigmoid(input).view(bs,-1)
    iflat = (iflat > 0.5)
    tflat = target.view(bs,-1)
    intersection = (iflat * tflat).sum(dim=1)
    
    return (2.0 * intersection + smooth) / (iflat.sum(dim=1) + tflat.sum(dim=1) + smooth)


class ConstantLR:
    def __init__(self):
        pass
    def step(self):
        pass

class AverageMetter:
    def __init__(self):
        self.step = 0
        self.value = 0
        self.sum = 0

    def update(self, value):
        self.sum += value
        self.step += 1
        self.value = self.sum/self.step 


def set_seed(seed, desterministic_algorithm=True):
    import torch
    import random
    import numpy as np


    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(desterministic_algorithm)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    return seed_worker, g

def log(x, file):
    print(x)
    with open(file, 'a') as f:
        f.write(x + '\n')