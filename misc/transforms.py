import numbers
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from config import cfg
import torch
import numpy
import pdb
import os
import torch.nn.functional as F
from PIL import ImageEnhance
import cv2
from torchvision.transforms import functional as TrF
from torchvision import transforms as tf
from misc import inflation

def exact_feature_distribution_matching(content, tf):
    tra_root = '/data/gjy/Datas/SHHA/images'
    tra_lst = os.listdir(tra_root)
    tra_img = np.random.randint(0, len(tra_lst))
    tra_img = Image.open(os.path.join(tra_root, tra_lst[tra_img])).convert('RGB')
    style = tf(tra_img)
    B, C, W, H = content.size(0), content.size(1), content.size(2), content.size(3)
 
    if not (content.size() == style.size()):
        style = F.interpolate(style, (W, H))

    _, index_content = torch.sort(content.view(B,C,-1))  ## sort content feature
    value_style, _ = torch.sort(style.view(B,C,-1))      ## sort style feature
    inverse_index = index_content.argsort(-1)
    transferred_content = content.view(B,C,-1) + value_style.gather(-1, inverse_index) - content.view(B,C,-1).detach()
    return transferred_content.view(B, C, W, H)

class ProcessSub(object):
    def __init__(self,T=0.1,K=51):
        self.T = T
        self.inf = inflation.inflation(K=K)

    def getHS(self,flow):
        # h direction  s or v magnitude
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        h = ang * 180 / np.pi / 2 #angle
        s = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)#magnitude
        return h,s

    def __call__(self, flow):
        h, s = self.getHS(flow[:, :, 0:2])
        flow[:,:, 0] = h.astype(np.float32) / 255
        flow[:,:, 1] = s.astype(np.float32) / 255
        # Threshold
        temp = np.ones(flow[:,:,2].shape)
        temp[abs(flow[:,:,2])<self.T] = 0
        flow[:,:,2] = flow[:,:,2] * temp
        # inflation
        return flow

# ===============================img tranforms============================

def Brightness(image):
    enh_bri = ImageEnhance.Brightness(image)
    brightness = np.random.randint(0, 20) / 10
    image = enh_bri.enhance(brightness)
    return image

def Chromaticity(image):
    enh_col = ImageEnhance.Color(image)
    color = np.random.randint(0, 20) / 10
    image = enh_col.enhance(color)
    return image

def Contrast(image):
    enh_con = ImageEnhance.Contrast(image)
    contrast = np.random.randint(0, 20) / 10
    image = enh_con.enhance(contrast)
    return image

def Sharpness(image):
    enh_sha = ImageEnhance.Sharpness(image)
    sharpness = np.random.randint(0, 30) / 10
    image = enh_sha.enhance(sharpness)
    return image

class RandomAugment(object):
    def __init__(self):
        self.Candidates = ['Brightness', 'Chromaticity', 'Contrast', 'Sharpness', 'EFDM', 'Noise']
        self.tf = tf.Compose([
            tf.ToTensor(),
            lambda x: x*255,
            lambda x: x.unsqueeze(0)
        ])
        self.re_tf = tf.Compose([
            lambda x: x.squeeze(0),
            lambda x: x.numpy(),
            lambda x: np.transpose(x, (1, 2, 0)),
            lambda x: Image.fromarray(x, mode='RGB')
        ])
        
    
    def __call__(self, img):
        
        num_process = np.random.randint(1, len(self.Candidates)+1)
        chosen_ones = np.random.choice(self.Candidates, num_process)
        specialize = False
        for one in chosen_ones:
            # print(type(img))
            if one not in ['EFDM', 'Noise']:
                img = eval(f'{one}(img)')
            else:
                specialize = True
        if specialize:
            img = self.tf(img)
            if 'EFDM' in self.Candidates:
                img = exact_feature_distribution_matching(img, self.tf)
            else:
                noise_ratio = 0.2
                noise = torch.clamp(torch.randn_like(img) * 0.1, -noise_ratio, noise_ratio)
                img = noise_ratio + img
            img = self.re_tf(img)
        return img
            

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, bbx=None):
        if bbx is None:
            for t in self.transforms:
                img,mask = t(img, mask)
            return img,mask
        for t in self.transforms:
            img,mask, bbx = t(img, mask, bbx)
        return img, mask, bbx

class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, bbx=None):
        if random.random() < 0.5:
            if bbx is None:
                # for i in range(3):
                #     flow[:,:,i] = np.fliplr(flow[:,:,i])
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)#, flow
            w, h = img.size
            xmin = w - bbx[:,3]
            xmax = w - bbx[:,1]
            bbx[:,1] = xmin
            bbx[:,3] = xmax
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), bbx
        if bbx is None:
            return img,mask  #flow
        return img, mask, bbx


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask, dst_size=None ):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        if dst_size is None:
            th, tw = self.size
        else:
            th, tw = dst_size
        if w == tw and h == th:
            return img, mask

        assert w >= tw
        assert h >= th

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        # flow = flow[y1:y1+th,x1:x1+tw,:]
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)) #.flow


class ScaleByRateWithMin(object):
    def __init__(self, rateRange, min_w, min_h):
        self.rateRange = rateRange
        self.min_w = min_w
        self.min_h = min_h
    def __call__(self, img, mask):# dot, flow):
        w, h = img.size
        # print('ori',w,h)
        rate = random.uniform(self.rateRange[0], self.rateRange[1])
        new_w = int(w * rate) // 32 * 32
        new_h = int(h * rate) // 32 * 32
        if new_h< self.min_h or new_w<self.min_w:
            if new_w<self.min_w:
                new_w = self.min_w
                rate = new_w/w
                new_h = int(h*rate) // 32*32
            if new_h < self.min_h:
                new_h = self.min_h
                rate = new_h / h
                new_w =int( w * rate) //32*32

        img = img.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)

        return img, mask 

# ===============================image tranforms============================

class RGB2Gray(object):
    def __init__(self, ratio):
        self.ratio = ratio  # [0-1]

    def __call__(self, img):
        if random.random() < 0.1:
            return  TrF.to_grayscale(img, num_output_channels=3)
        else: 
            return img

class GammaCorrection(object):
    def __init__(self, gamma_range=[0.4,2]):
        self.gamma_range = gamma_range 

    def __call__(self, img):
        if random.random() < 0.5:
            gamma = random.uniform(self.gamma_range[0],self.gamma_range[1])
            return  TrF.adjust_gamma(img, gamma)
        else: 
            return img

# ===============================label tranforms============================

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()




# class tensormul(object):
#     def __init__(self, mu=255.0):
#         self.mu = 255.0
    
#     def __call__(self, _tensor):
#         _tensor.mul_(self.mu)
#         return _tensor
