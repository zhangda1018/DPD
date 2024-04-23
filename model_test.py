import torch, os
from model.HRSelector.HRSelector import *
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = '2'




def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()


pred = torch.ones((1, 1, 32, 32))

gt = torch.ones((1, 1, 32, 32))

print(dice_loss(pred, gt))





