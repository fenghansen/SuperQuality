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


def log(string):
    print(time.strftime('%H:%M:%S'), ">> ", string)

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


# def rgb2hsv(img):
#     if torch.is_tensor:
#         log(f'Image tensor size is {img.size()}')
#     else:
#         log("This Function can only deal PyTorch Tensor!")
#         return img
#     r, g, b = img.split(1, 0)
#     tensor_max = torch.max(torch.max(r, g), b)
#     tensor_min = torch.min(torch.min(r, g), b)
#     m = tensor_max-tensor_min
#     if tensor_max == tensor_min:
#         h = 0
#     elif tensor_max == r:
#         if g >= b:
#             h = ((g-b)/m)*60
#         else:
#             h = ((g-b)/m)*60 + 360
#     elif tensor_max == g:
#         h = ((b-r)/m)*60 + 120
#     elif tensor_max == b:
#         h = ((r-g)/m)*60 + 240
#     if tensor_max == 0:
#         s = 0
#     else:
#         s = m/tensor_max
#     v = tensor_max
#     return h, s, v


def sample(imgs, split=None ,figure_size=(2, 3), img_dim=(400, 600), path=None, num=0):
    if type(img_dim) is int:
        img_dim = (img_dim, img_dim)
    img_dim = tuple(img_dim)
    if len(img_dim) == 1:
        h_dim = img_dim
        w_dim = img_dim
    elif len(img_dim) == 2:
        h_dim, w_dim = img_dim
    h, w = figure_size
    if split is None:
        num_of_imgs = figure_size[0] * figure_size[1]
        gap = len(imgs) // num_of_imgs
        split = list(range(0, len(imgs)+1, gap))
    figure = np.zeros((h_dim*h, w_dim*w, 3))
    for i in range(h):
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
        log("Saving Image at {}".format(path))
        cv2.imwrite(path, figure)