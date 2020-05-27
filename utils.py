import os
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import collections
import torch
import torchvision
import shutil
import time
from skimage.measure import compare_psnr, compare_ssim

def log(string, log=None):
    log_string = f'{time.strftime("%H:%M:%S")} >>  {string}'
    print(log_string)
    if log is not None:
        with open(log,'a+') as f:
            f.write(log_string+'\n')

def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)

# 作为装饰器函数
def no_grad(fn):
    with torch.no_grad():
        def transfer(*args,**kwargs):
            fn(*args,**kwargs)
        return fn


def load_weights(model, path):
    pretrained_dict=torch.load(path)
    model_dict=model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # k1 = 'conv1.doubleconv.0.conv_relu.0.weight'
    # k2 = 'conv1.doubleconv.0.conv_relu.0.bias'
    # if k1 in pretrained_dict:
    #     del pretrained_dict[k1], pretrained_dict[k2]
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()


class EMA():
    '''
    # 转自知乎https://zhuanlan.zhihu.com/p/68748778
    # 初始化
    ema = EMA(model, 0.999)
    ema.register()

    # 训练过程中，更新完参数后，同步update shadow weights
    def train():
        optimizer.step()
        ema.update()

    # eval前，apply shadow weights；eval之后，恢复原来模型的参数
    def evaluate():
        ema.apply_shadow()
        # evaluate
        ema.restore()
    '''
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
        

def rgb2hsv(img, dim=0):
    if torch.is_tensor:
        log(f'Image tensor size is {img.size()}')
    else:
        log("This Function can only deal PyTorch Tensor!")
        return img
    # img = img * 0.5 + 0.5
    # hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

    # hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + 1e-7) ) [ img[:,2]==img.max(1)[0] ]
    # hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + 1e-7) ) [ img[:,1]==img.max(1)[0] ]
    # hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + 1e-7) ) [ img[:,0]==img.max(1)[0] ]) % 6

    # hue[img.min(1)[0]==img.max(1)[0]] = 0.0
    # hue = hue/6

    # saturation = ( img.max(1)[0] - img.min(1)[0] ) / ( img.max(1)[0] + 1e-7 )
    # saturation[ img.max(1)[0]==0 ] = 0

    # value = img.max(1)[0]
    # img_hsv = torch.cat([hue.view(1,1,400,600),saturation.view(1,1,400,600),value.view(1,1,400,600)], dim=dim)
    # return img_hsv
    r, g, b = img.split(1, dim=dim)
    gap = 1./6.
    H = torch.zeros_like(r)
    S = torch.zeros_like(r)
    tensor_max = torch.max(torch.max(r, g), b)
    tensor_min = torch.min(torch.min(r, g), b)
    delta = tensor_max-tensor_min
    V = tensor_max
    g_b = (g >= b)
    b_g = ~g_b
    H_r_g = ((g-b)/delta)*gap
    H_r_b = ((g-b)/delta)*gap + 1.
    H_b = ((b-r)/delta)*gap + 2*gap
    H_g = ((r-g)/delta)*gap + 4*gap
    H_test = H.numpy().transpose(1,2,0)
    cv2.imshow('hsv1', H_test)
    cv2.waitKey(0)
    cv2.destroyWindow('hsv1')
    log(tensor_max==r)
    log(tensor_max==g)
    log(tensor_max==b)
    log(g_b)
    log(b_g)
    H = torch.where((tensor_max==r) & g_b, H, H_r_g)
    H_test = H.numpy().transpose(1,2,0)
    cv2.imshow('hsv1', H_test)
    cv2.waitKey(0)
    cv2.destroyWindow('hsv1')
    H = torch.where((tensor_max==r) & b_g, H, H_r_b)
    H_test = H.numpy().transpose(1,2,0)
    cv2.imshow('hsv1', H_test)
    cv2.waitKey(0)
    cv2.destroyWindow('hsv1')
    H = torch.where(tensor_max==g, H, H_b)
    H_test = H.numpy().transpose(1,2,0)
    cv2.imshow('hsv1', H_test)
    cv2.waitKey(0)
    cv2.destroyWindow('hsv1')
    H = torch.where(tensor_max==g, H, H_g)
    H_test = H.numpy().transpose(1,2,0)
    cv2.imshow('hsv1', H_test)
    cv2.waitKey(0)
    cv2.destroyWindows()
    S = torch.where(tensor_max!=0, S, delta / tensor_max)
    V = tensor_max
    img_hsv = torch.cat([H,S,V], dim=dim)
    return img_hsv

def hsv2rgb(img, dim=0):
    if torch.is_tensor:
        log(f'Image tensor size is {img.size()}')
    else:
        log("This Function can only deal PyTorch Tensor!")
        return img
    h, s, v = img.split(1, dim=dim)
    h60 = h * 60
    h60f = torch.floor(h60)
    hi = h60f % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    vtp = torch.cat([v,t,p], dim=dim)
    qvp = torch.cat([q,v,p], dim=dim)

    rgb = torch.zeros_like(img)
    rgb = torch.where(hi==0, vtp, rgb)
    rgb = torch.where(hi==1, qvp, rgb)
    rgb = torch.where(hi==2, vtp[2,0,1], rgb)
    rgb = torch.where(hi==3, qvp[2,0,1], rgb)
    rgb = torch.where(hi==4, vtp[1,2,0], rgb)
    rgb = torch.where(hi==5, qvp[1,2,0], rgb)
    # if hi == 0: r, g, b = v, t, p
    # elif hi == 1: r, g, b = q, v, p
    # elif hi == 2: r, g, b = p, v, t
    # elif hi == 3: r, g, b = p, q, v
    # elif hi == 4: r, g, b = t, p, v
    # elif hi == 5: r, g, b = v, p, q
    img_rgb = rgb
    return img_rgb


def sample(imgs, split=None ,figure_size=(1, 1), img_dim=(400, 600), path=None, num=0, metrics=False):
    if type(img_dim) is int:
        img_dim = (img_dim, img_dim)
    img_dim = tuple(img_dim)
    if len(img_dim) == 1:
        h_dim = img_dim
        w_dim = img_dim
    elif len(img_dim) == 2:
        h_dim, w_dim = img_dim
    h, w = figure_size
    num_of_imgs = figure_size[0] * figure_size[1]
    gap = len(imgs) // num_of_imgs
    colormap = True# if gap > 1 else False
    if split is None:
        split = list(range(0, len(imgs)+1, gap))
    figure = np.zeros((h_dim*h, w_dim*w, 3))
    for i in range(h):
        if metrics:
            img_hr = imgs[ split[i*w+w-1] : split[i*w+w] ].transpose(1,2,0)
        for j in range(w):
            idx = i*w+j
            if idx >= len(split)-1: break
            digit = imgs[ split[idx] : split[idx+1] ]
            if len(digit) == 1:
                for k in range(3):
                    figure[i*h_dim: (i+1)*h_dim,
                        j*w_dim: (j+1)*w_dim, k] = digit
            elif len(digit) == 3:
                for k in range(3):
                    figure[i*h_dim: (i+1)*h_dim,
                        j*w_dim: (j+1)*w_dim, k] = digit[2-k]
            if metrics:
                if j < w-1: 
                    img_lr = digit.transpose(1,2,0)
                    psnr = compare_psnr(img_hr, img_lr)
                    ssim = compare_ssim(img_hr, img_lr, multichannel=colormap)
                    text = f'psnr:{psnr:.4f} - ssim:{ssim:.4f}'
                    log(text, log='./logs.txt')
                    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                    cv2.putText(figure, text, (j*w_dim, i*h_dim+20), font, 1, (255,255,255))
                else: 
                    text = 'reference'
                    log(text)
                    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                    cv2.putText(figure, text, (j*w_dim, i*h_dim+20), font, 1, (255,255,255))

    if path is None:
        cv2.imshow('Figure%d'%num, figure)
        cv2.waitKey()
    else:
        figure *= 255
        filename1 = path.split('\\')[-1]
        filename2 = path.split('/')[-1]
        if len(filename1) < len(filename2):
            filename = filename1
        else:
            filename = filename2
        root_path = path[:-len(filename)]
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        log("Saving Image at {}".format(path), log='./logs.txt')
        cv2.imwrite(path, figure)
    