import torch
import numpy as np
import sys
sys.path.append('/vinbrain/huyta/segmentation')
from TedSeg import *
from .fda import create_fda_batch
from .dataset import *
from .fda import *
from .unet import *
from .losses import *

@torch.no_grad()
def prepare_similarity_label(y, indices):
    y = y[indices]*y
    y[y==0] = -1
    return y.contiguous()

@torch.no_grad()
def min_max_scaler(x):
    x_flat = x.flatten(start_dim=1)
    x_min = x_flat.min(1, keepdim=True).values.unsqueeze(-1).unsqueeze(-1)
    x_max = x_flat.max(1, keepdim=True).values.unsqueeze(-1).unsqueeze(-1)
    x = (x - x_min)/(x_max - x_min)
    return x

@torch.no_grad()
def prepare_batches_for_training(xs, ys, xt, yt, beta, device='cuda:0'):
    bs = xs.shape[0]
    
    ones = torch.ones([bs//2])
    zeros = torch.zeros([bs//2])
    
    indices = np.arange(bs)
    idx_top = indices[:bs//2]
    idx_bot = indices[bs//2:]
    idx_mid = indices[bs//4:-bs//4]
    
    xs_to_t = create_fda_batch(fr=xs[idx_top], to=xt[idx_bot], beta=beta)
    xt_to_s = create_fda_batch(fr=xt[idx_top], to=xs[idx_bot], beta=beta)
    
    x = torch.cat([xs[idx_mid], xt[idx_mid], xs_to_t, xt_to_s], dim=0).to(device)
    y = torch.cat([ys[idx_mid], yt[idx_mid],  yt[idx_bot], ys[idx_bot]], dim=0).to(device)
    
    # 0: source, 1:target
    style_label = torch.cat([zeros, ones, zeros, ones]).to(device)
    content_label = torch.cat([zeros, ones, ones, zeros]).to(device)

    
    return {
        'img': (x/255).contiguous(),
        'mask': y.contiguous(),
        'style': style_label.contiguous(),
        'content': content_label.contiguous()
    }

class CoFoLearner(Learner):
    def fit(self, n_epochs, optim, lr_scheduler, cofo_loss, beta=0.09, contrastive=True, alpha=1.0, alpha_content=1.0,
        schedule_alpha=True, scheduler_mode='iteration', validate_every=10):

        train_dl, valid_dl = self.data

        # schedule alpha
        if schedule_alpha:
            max_steps = len(train_dl)*n_epochs
            step = 0

        bar = master_bar(range(n_epochs))
        for epoch in bar:
            # train
            self.model.train()
            self.train_dice_loss = AverageMetter()
            self.train_style_loss = AverageMetter()
            self.train_content_loss = AverageMetter()
            train_bar = progress_bar(train_dl,parent=bar)
            for xs, ys, xt, yt in train_bar:
                optim.zero_grad()
                xs, ys, xt, yt = xs.cuda(), ys.cuda(), xt.cuda(), yt.cuda()

                batches = prepare_batches_for_training(xs, ys, xt, yt, beta)

                # train segmentation
                x = batches['img']
                y = batches['mask']

                pred_mask, feature_1 = self.model(x, mode='feature')
                loss_segmentation = cofo_loss['segmentation'](pred_mask, y, batches['content'])
                loss = loss_segmentation

                # train contrastive
                if contrastive:
                    bs = x.shape[0]
                    indices = np.arange(bs)
                    np.random.shuffle(indices)

                    feature_2 = feature_1[indices].clone()

                    style_feature_1, content_feature_1 = self.model(feature_1, mode='contrastive')
                    style_feature_2, content_feature_2 = self.model(feature_2, mode='contrastive')

                    style_sim_label = prepare_similarity_label(batches['style'], indices)
                    content_sim_label = prepare_similarity_label(batches['content'], indices)

                    loss_style = cofo_loss['contrastive'](style_feature_1, style_feature_2, style_sim_label)
                    loss_content = cofo_loss['contrastive'](content_feature_1, content_feature_2, content_sim_label)

                    if schedule_alpha:
                        p = step/max_steps
                        alp = 2/(1 + np.exp(-10*p)) - 1
                        step += 1
                        alpha = alpha*alp
                        alpha_content = alpha_content*alp

                    loss += alpha*loss_style + alpha_content*loss_content
                
                loss.backward()

                self.train_dice_loss.update(loss_segmentation.item())
                if contrastive:
                    self.train_style_loss.update(loss_style.item())
                    self.train_content_loss.update(loss_content.item())
                train_bar.comment = f'train_dice_loss: {self.train_dice_loss.value:0.6f}'

                optim.step()
                if scheduler_mode == 'iteration':
                    lr_scheduler.step()

            self.save()
            self.show_img(batches, bar)

            if epoch % validate_every == 0:
                val_loss, val_metric = self.validate(valid_dl)

                if scheduler_mode == 'validation':
                    lr_scheduler.step(val_loss)

                saved = ''
                if self.valid_loss.value < self.best_valid_loss:
                    self.best_valid_loss = self.valid_loss.value
                    self.save(mode='best')
                    saved = '-saved'
                
                log = f'Ep. {epoch:>4}|' + f'train_dice_loss: {self.train_dice_loss.value:.3f}|'
                if contrastive:
                    log += f'train_style_loss: {self.train_style_loss.value:.3f}|'
                    log += f'train_content_loss: {self.train_content_loss.value:.3f}|'

                log += f'valid_loss: {self.valid_loss.value:.3f}|'
                log += f'valid_metric: {self.valid_metric:.3f} {saved}'

                self.log(log)

    @torch.no_grad()
    def show_training_img(self, batches):
        img_grid = batches['img'].cpu()
        b,c,h,w = img_grid.shape
        img_grid = img_grid.view(4, -1, c, h, w)[:,0]
        img_grid = min_max_scaler(img_grid)
        img_grid = torch.cat(list(img_grid), dim=2)
        

        msk_grid = batches['mask'].cpu()
        b,c,h,w = msk_grid.shape
        msk_grid = msk_grid.view(4, -1, c, h, w)[:,0]
        msk_grid = torch.cat(list(msk_grid), dim=2)

        return (CoFoSegmentationDataset.create_blend(img_grid, msk_grid, denormalize=False)*255).astype(np.uint8)

    @torch.no_grad()
    def show_img(self, idx=None, batches=None, bar=None, verbose=False):
        ds = self.data[1].dataset
        self.model.eval()
        if idx is None:
            idx = np.random.randint(len(ds))
        if verbose:
            print(idx)
        img, mask = ds[idx]

        pred = self.model(img.unsqueeze(0).cuda())[0].detach().cpu()
        pred = torch.sigmoid(pred).numpy()
        img_pred = (pred[0]>=0.5)

        img = img.permute(1,2,0).detach().numpy()
        color_mask = np.array([0.4*img_pred, 0.0*img_pred, 0.92*img_pred])
        color_mask = np.transpose(color_mask, (1,2,0))
        img_pred = img_pred[...,None]

        pred_blend = 0.4*color_mask + 0.6*img*img_pred + (1 - img_pred)*img


        mask = mask.permute(1,2,0)[...,0].detach().numpy()
        color_mask = np.array([0.4*mask, 1.0*mask, 0.25*mask])
        color_mask = np.transpose(color_mask, (1,2,0))
        mask = mask[...,None]
        target_blend = 0.5*color_mask + 0.5*img*mask + (1 - mask)*img

        img = np.concatenate([img, target_blend, pred_blend, np.zeros_like(img)], axis=1)*255

        if batches is not None:
            training_img = self.show_training_img(batches)
            img = np.concatenate([training_img, img], axis=0)

        imgs_out = Image.fromarray(img.astype(np.uint8), 'RGB')

        if bar is None:
            display(imgs_out)
            return

        if not hasattr(bar, 'imgs_out'):
            bar.imgs_out = display(imgs_out, display_id=True)
        else: 
            bar.imgs_out.update(imgs_out)