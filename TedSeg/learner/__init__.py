import os
from fastprogress import  master_bar, progress_bar
from PIL import Image
import numpy as np
from numpy.random import choice
from matplotlib import pyplot as plt
from IPython.display import display
import torch
from functools import partial
from ..utils import AverageMetter, log
     

class Learner:
    def __init__(self, model, data, loss, metric=None, name='model', model_path='./weights'):
        self.model = model.cuda()
        self.data = data # (train_dl, valid_dl)
        self.loss = loss
        self.name = name
        self.metric = metric
        self.best_valid_loss = 100
        self.model_path = model_path
        
        log_path = './' + name + '.txt'
        self.log = partial(log, file=log_path)
        with open(log_path, 'w') as f:
            f.write(self.name + '\n')

    def save(self, mode=None):
        os.makedirs(self.model_path, exist_ok=True)
        if mode is None:
            mode = ''
        else:
            mode = '-' + mode
        torch.save(self.model.state_dict(), f'{self.model_path}/{self.name}{mode}.pth')
    
    
    def fit(self, n_epochs, optim, lr_scheduler, scheduler_mode='iteration', validate_every=10):
        train_dl, valid_dl = self.data
        
        bar = master_bar(range(n_epochs))
        for epoch in bar:
            # train
            self.model.train()
            self.train_loss = AverageMetter()
            train_bar = progress_bar(train_dl,parent=bar)
            for xb, yb in train_bar:
                optim.zero_grad()
                xb, yb = xb.cuda(), yb.cuda()
                pred = self.model(xb)
                loss = self.loss(pred, yb)
                loss.backward()
                self.train_loss.update(loss.item())
                train_bar.comment = f'train_loss: {self.train_loss.value:0.6f}'
                
                optim.step()

                if scheduler_mode == 'iteration':
                    lr_scheduler.step()

            self.save()
            self.show_img(bar)

            if epoch % validate_every == 0:
                val_loss, val_metric = self.validate(valid_dl)
                
                if scheduler_mode == 'validation':
                    lr_scheduler.step(val_loss)
                    
                saved = ''
                if self.valid_loss.value < self.best_valid_loss:
                    self.best_valid_loss = self.valid_loss.value
                    self.save(mode='best')
                    saved = '-saved'
                self.log(f'Ep. {epoch:>4}|train_loss: {self.train_loss.value:0.4f}|valid_loss: {self.valid_loss.value:0.4f}|valid_metric: {self.valid_metric:.3f} {saved}')

        self.save(mode='last')
    
    @torch.no_grad()
    def validate(self, valid_dl):
        self.model.eval()
        self.valid_loss = AverageMetter()
        dices = []
        for xb, yb in valid_dl:
            xb, yb = xb.cuda(), yb.cuda()
            pred = self.model(xb).detach()
            loss = self.loss(pred, yb)
            if self.metric is not None:
                dices.append(self.metric(pred, yb).detach().cpu().numpy())
            else:
                dices.append([-1])
            self.valid_loss.update(loss.item())
        self.valid_metric = np.concatenate(dices).mean()
        return self.valid_loss.value, self.valid_metric
            

from .utils import *
