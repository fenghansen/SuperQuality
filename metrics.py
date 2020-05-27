import os
import cv2
import numpy as np
from skimage.measure import compare_psnr, compare_ssim
from utils import log
from NIQE import niqe

if __name__ == "__main__":
    psnrs = []
    ssims = []
    NIQEs = []
    lr_dir='./images/compare-lr-custom-best'
    hr_dir='./images/compare-hr'
    out_dir = r'H:\datasets\SIDD\SIDD_Medium_Srgb\final_output'
    ones = np.ones((400, 600), dtype=np.float32)
    lr_paths = [os.path.join(lr_dir,name) for name in os.listdir(lr_dir)]
    hr_paths = [os.path.join(hr_dir,name) for name in os.listdir(hr_dir)]
    for lr_path, hr_path in zip(lr_paths, hr_paths):
        lr_img = cv2.imread(lr_path)
        hr_img = cv2.imread(hr_path)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2HSV)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2HSV)
        # cv2.imshow('lr1',lr_img)
        # cv2.imshow('hr1',hr_img)
        lr_img[:,:,-1] = ones*255
        hr_img[:,:,-1] = ones*255
        # cv2.imshow('lr2',lr_img)
        # cv2.imshow('hr2',hr_img)
        
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_HSV2BGR)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_HSV2BGR)

        psnr = compare_psnr(hr_img, lr_img)
        ssim = compare_ssim(hr_img, lr_img, multichannel=True)
        # NIQE = niqe(hr_img)
        # NIQEs.append(NIQE)
        psnrs.append(psnr)
        ssims.append(ssim)
        # cv2.imwrite(hr_path[:-4]+'_hsv.png', hr_img)
        # cv2.imshow('lr',lr_img)
        # cv2.imshow('hr',hr_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite(path[:-4]+'crop.png', crop)
        # cv2.imwrite(path[:-4]+'bigcrop.png', bigcrop)
    log(psnrs,log=lr_dir+'/score.txt')
    # log(ssims,log=lr_dir+'/score.txt')
    # log(NIQEs,log=lr_dir+'/score.txt')
    log(f"hsv_psnr:{np.mean(psnrs):.2f} - ssim:{np.mean(ssims):.4f}",log=lr_dir+'/score.txt')
    # log(f"psnr:{np.mean(psnrs):.2f} - ssim:{np.mean(ssims):.4f}",log=lr_dir+'/score.txt')
    # log(f"psnr:{np.mean(psnrs):.2f} - ssim:{np.mean(ssims):.4f} - NIQE:{np.mean(NIQEs):.4f}",log=lr_dir+'/score.txt')