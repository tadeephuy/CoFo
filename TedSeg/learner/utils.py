from . import *
from ..utils import *

@torch.no_grad()
def show_img(self, bar=None):
    ds = self.data[1].dataset
    self.model.eval()
    idx = np.random.randint(len(ds))
    img, mask = ds[idx]
    
    pred = self.model(img.unsqueeze(0).cuda())[0].detach().cpu()
    pred = torch.sigmoid(pred).numpy()
    img_pred = (pred[0]>=0.5)

    img = img.permute(1,2,0).detach().numpy()
    color_mask = np.array([0.4*img_pred, 0.0*img_pred, 0.92*img_pred])
    color_mask = np.transpose(color_mask, (1,2,0))
    img_pred = img_pred[...,None]

    img = (img * 0.22) + 0.5
    pred_blend = 0.4*color_mask + 0.6*img*img_pred + (1 - img_pred)*img


    mask = mask.permute(1,2,0)[...,0].detach().numpy()
    color_mask = np.array([0.4*mask, 1.0*mask, 0.25*mask])
    color_mask = np.transpose(color_mask, (1,2,0))
    mask = mask[...,None]
    target_blend = 0.5*color_mask + 0.5*img*mask + (1 - mask)*img

    img = np.concatenate([img, target_blend, pred_blend], axis=1)*255
    imgs_out = Image.fromarray(img.astype(np.uint8), 'RGB')

    if bar is None:
        display(imgs_out)
        return

    if not hasattr(bar, 'imgs_out'):
        bar.imgs_out = display(imgs_out, display_id=True)
    else: 
        bar.imgs_out.update(imgs_out)
Learner.show_img = show_img

def get_dice_score(self, test_loader):
    self.model.eval()
    dices = []
    with torch.no_grad():
        b = progress_bar(test_loader)
        for xb, yb in b:
            xb = xb.cuda()
            yb = yb.cuda()
            pred = self.model(xb)
            dice = batch_dice_score(pred, yb)
            dice = dice.detach().cpu().numpy()
            dices.append(dice)
            dice_score = np.concatenate(dices, axis=0)
            b.comment = f'{dice_score.mean():.2f}'
    return dice_score.mean()
Learner.get_dice_score = get_dice_score


@torch.no_grad()
def predict(self, img_path, preprocess=None):
    self.model.eval()
    img = self.data[1].dataset.imread(img_path)
    if preprocess is None:
        img = self.data[1].dataset.transforms(image=img)['image']
    else:
        img = preprocess(img)
    img = torch.tensor(np.transpose(img, (2,0,1)))/255
    if self.data[1].dataset.normalize:
        img = (img - 0.5)/0.22
    img = img[None].cuda()
    mask = self.model(img).sigmoid().detach().cpu().numpy()[0]
    return mask
Learner.predict = predict