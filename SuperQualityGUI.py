import tkinter as tk
import numpy as np
import os
import cv2
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename
from PIL import Image, ImageTk
import torch
import time
import yaml
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision.utils import make_grid
from torchvision import transforms
from torchsummary import summary
from base_trainer import BaseTrainer
from losses import *
from models import *
from base_parser import BaseParser
from dataloader import *

top_source = None
top_output = None

label_source = None
label_output = None

img_source = None
pimg_source = None
img_output = None
pimg_output = None

ratio = 1.4
exposure = 80
light = 80

L_input_np = None
R_decom_np = None
I_decom_np = None
R_final_np = None
I_final_np = None
output_np = None


class KinD_GUI(BaseTrainer):
    def __init__(self, model):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device=self.device)

    def compute(self):
        global I_decom_np
        global R_decom_np
        global R_final_np
        global L_input_np
        self.model.eval()
        self.model.to(device=self.device)
        L_low_tensor = torch.from_numpy(L_input_np)
        L_low = L_low_tensor.to(self.device)
        with torch.no_grad():
            # 连续亮度过渡
            R_low, I_low = self.model.decom_net(L_low)
            R_final = self.model.restore_net(R_low, I_low)
            R_final_np = R_final.detach().cpu().numpy()[0]
            R_decom_np = R_low.detach().cpu().numpy()[0]
            I_decom_np = I_low.detach().cpu().numpy()[0]

        do_show()


def set_None_top_source():
    global top_source
    top_source.destroy()
    top_source = None

def set_None_top_output():
    global top_output
    top_output.destroy()
    top_output = None


def open_img():
    global img_source
    global pimg_source
    global top_source
    global label_source
    global L_input_np

    img_path = askopenfilename()
    if os.path.exists(img_path):
        img_source = Image.open(img_path).convert('RGB')
        pimg_source = ImageTk.PhotoImage(img_source)
        L_input_np = np.array(img_source).astype(np.float32)/255.
        if top_source is None:
            top_source = tk.Toplevel()
            top_source.title('Source Image')
            top_source.protocol('WM_DELETE_WINDOW', set_None_top_source)
            label_source = tk.Label(top_source)
            label_source.pack()
        label_source.configure(image=pimg_source)

def do_show():
    global img_source
    global img_output
    global pimg_output
    global top_output
    global label_output
    global I_decom_np
    global R_decom_np
    global I_final_np
    global R_final_np
    global output_np
    global ratio
    global exposure
    global light

    if img_source is None:
        return
    if I_decom_np is None:
        log(f'You need to click "compute" first!!')
    
    ratio = scale_ratio.get()
    exposure = scale_exposure.get()
    light = scale_light.get()

    log(f'ratio:{ratio}, exposure:{exposure}, light:{light}')

    # torch
    I_low = torch.from_numpy(I_decom_np).to('cuda')
    R_low = torch.from_numpy(R_decom_np).to('cuda')
    R_final = torch.from_numpy(R_final_np).to('cuda')
    with torch.no_grad():
        I_final, I_standard = self.model.illum_net(I_low, ratio)
        I_att = torch.clamp(F.sigmoid((0.5-I_low)*10)/0.99330, min=1-exposure*0.01, max=1.0)
        R_out = R_low + (R_final-R_low) * torch.cat([I_att, I_att, I_att], dim=1)
        I_step = 0.01*light*I_final + (1-0.01*light)*I_standard
        I_out = torch.cat([I_step, I_step, I_step], dim=1)
        output = I_out * R_out
    
    I_final_np = I_final.detach().cpu().numpy()[0]
    output_np = output.detach().cpu().numpy()[0]

    img_output = Image.fromarray(np.uint8(output_np*255))
    pimg_output = ImageTk.PhotoImage(img_output)

    if top_output is None:
        top_output = tk.Toplevel()
        top_output.title('Output Image')
        top_output.protocol('WM_DELETE_WINDOW', set_None_top_output)
        label_output = tk.Label(top_output)
        label_output.pack()
    top_output.deiconify()
    label_output.configure(image=pimg_output)

def do_save():
    global img_output
    global L_input_np
    global I_decom_np
    global R_decom_np
    global I_final_np
    global R_final_np
    global output_np

    # print(img_output)
    if img_output is None:
        return
    save_path = asksaveasfilename()
    # print(save_path)
    if save_path:
        img_output.save(save_path)
        save_more = save_more.get()
        if save_more:
            sample_imgs = np.concatenate( (R_decom_np, I_decom_np, L_input_np,
                                        R_final_np, I_final_np, output_np), axis=0 )
            filepath = f'{save_path[:-4]}_extra.png'
            split_point = [0, 3, 4, 7, 10, 11, 14]
            img_dim = I_decom_np.shape[1:]
            sample(sample_imgs, split=split_point, figure_size=(2, 3), 
                        img_dim=img_dim, path=filepath)

class TestParser(BaseParser):
    def parse(self):
        self.parser.add_argument("-p", "--plot_more", default=True,
                                help="Plot intermediate variables. such as R_images and I_images")
        self.parser.add_argument("-c", "--checkpoint", default="./weights/", 
                                help="Path of checkpoints")
        self.parser.add_argument("-i", "--input_dir", default="./images/inputs-light/", 
                                help="Path of input pictures")
        self.parser.add_argument("-o", "--output_dir", default="./images/outputs-illum-custom-light/", 
                                help="Path of output pictures")
        self.parser.add_argument("-r", "--ratio", default=1.4, help="Target brightness")
        self.parser.add_argument("-u", "--use_gpu", default=True, 
                                help="If you want to use GPU to accelerate")
        return self.parser.parse_args()


if __name__ == "__main__":
    # 加载torch模型
    model = KinD(use_MaskMul=True)
    parser = TestParser()
    args = parser.parse()

    input_dir = args.input_dir
    output_dir = args.output_dir
    plot_more = args.plot_more
    checkpoint = args.checkpoint
    decom_net_dir = os.path.join(checkpoint, "decom_net_normal.pth")
    restore_net_dir = os.path.join(checkpoint, "restore_GAN_mask.pth")
    illum_net_dir = os.path.join(checkpoint, "illum_net_custom_final.pth")
    
    model.decom_net = load_weights(model.decom_net, path=decom_net_dir)
    log('Model loaded from decom_net.pth')
    model.restore_net = load_weights(model.restore_net, path=restore_net_dir)
    log('Model loaded from restore_net.pth')
    model.illum_net = load_weights(model.illum_net, path=illum_net_dir)
    log('Model loaded from illum_net.pth')

    log("Buliding Dataset...")
    dst = CustomDataset(input_dir)
    log(f"There are {len(dst)} images in the input direction...")
    dataloader = DataLoader(dst, batch_size=1)

    KinD = KinD_GUI(model)

    # TK部分
    top = tk.Tk()
    top.title('SuperQuality')

    real_time = tk.BooleanVar()
    save_more = tk.BooleanVar()

    scale_ratio = tk.Scale(top, label='bright ratio', from_=0, to=2, resolution=0.05, orient=tk.HORIZONTAL, length=300)
    scale_exposure = tk.Scale(top, label='restore level(%)', from_=0, to=100, orient=tk.HORIZONTAL, length=300)
    scale_light = tk.Scale(top, label='light->standard(%)', from_=0, to=100, orient=tk.HORIZONTAL, length=300)
    check_real_time = tk.Checkbutton(top, text="real_time", variable=real_time, onvalue=True, offvalue=False)
    check_save_more = tk.Checkbutton(top, text="save_more", variable=save_more, onvalue=True, offvalue=False)

    scale_ratio.set(1.4)
    scale_exposure.set(80)
    scale_light.set(80)
    check_real_time.select()

    button_open = tk.Button(top, text='Open', command=open_img)
    button_compute = tk.Button(top, text='Compute', command=KinD.compute)
    button_save = tk.Button(top, text='Save as', command=do_save)

    scale_ratio.grid(row=0, column=0, columnspan=4)
    scale_exposure.grid(row=1, column=0, columnspan=4)
    scale_light.grid(row=2, column=0, columnspan=4)
    check_real_time.grid(row=3, column=0)
    check_save_more.grid(row=3, column=1)
    button_open.grid(row=4, column=0)
    button_compute.grid(row=4, column=1)
    button_save.grid(row=4, column=2)

    top.mainloop()
