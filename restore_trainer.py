import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import yaml
import sys
from torch.optim import lr_scheduler
from torchvision import transforms
from torchsummary import summary
from base_trainer import BaseTrainer
from base_parser import BaseParser
from losses import *
from models import *
from dataloader import *

class Restore_Trainer(BaseTrainer):
    def __init__(self, config, dataloader, criterion, model, 
            dataloader_test=None, decom_net=None):
        super().__init__(config, dataloader, criterion, model, dataloader_test)
        log(f'Using device {self.device}')
        self.decom_net = decom_net
        self.decom_net.to(device=self.device)

    def train(self):
        # print(self.model)
        summary(self.model, input_size=[(3, 256, 256), (1,256,256)])

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.997) #0.977237, 0.986233
        try:
            for iter in range(self.epochs):
                self.model.train()
                epoch_loss = 0
                idx = 0
                hook_number = -1
                iter_start_time = time.time()
                gpu_time = 0

                for L_low_tensor, L_high_tensor, name in self.dataloader:
                    optimizer.zero_grad()
                    L_low = L_low_tensor.to(self.device)
                    L_high = L_high_tensor.to(self.device)

                    gpu_time_iter = time.time()
                    with torch.no_grad():
                        R_low, I_low = self.decom_net(L_low)
                        R_high, I_high = self.decom_net(L_high)
                        
                    # R_restore = self.model(R_low, I_low)
                    R_restore,R2,R4,R8 = self.model(R_low, I_low, mode='train')

                    if idx % self.print_frequency == 0:
                        hook_number = iter
                    loss = self.loss_fn(R_restore, R_high,R2,R4,R8, hook=hook_number)
                    hook_number = -1
                    if idx % 8 == 0:
                        with torch.no_grad():
                            psnr = PSNR_Loss(R_restore, R_high)
                        log(f"iter: {iter}_{idx}\taverage_loss: {loss.item():.6f} - PSNR:{psnr:.2f}")
                    loss.backward()
                    optimizer.step()
                    gpu_time += time.time()-gpu_time_iter 
                    idx += 1

                if iter % self.print_frequency == 0:
                    self.test(iter, plot_dir='./images/samples-restore-mask-finetune')

                if iter % self.save_frequency == 0:
                    torch.save(self.model.state_dict(), f'./weights/restore-finetune/restore_mask_finetune_{iter//100}.pth')
                    log("Weight Has saved as 'restore_net.pth'")
                
                scheduler.step()
                iter_end_time = time.time()
                log(f"Time taken: {iter_end_time - iter_start_time:.3f} sec, include gpu_time {gpu_time:.3f} sec\t lr={scheduler.get_lr()[0]:.6f}")

        except KeyboardInterrupt:
            torch.save(self.model.state_dict(), './weights/INTERRUPTED_restore.pth')
            print('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

    @no_grad
    def test(self, epoch=-1, plot_dir='./images/samples-restore'):
        self.model.eval()
        for L_low_tensor, L_high_tensor, name in self.dataloader_test:
            L_low = L_low_tensor.to(self.device)
            L_high = L_high_tensor.to(self.device)

            R_low, I_low = self.decom_net(L_low)
            R_high, I_high = self.decom_net(L_high)

            R_restore = self.model(R_low, I_low)
            L_output = R_restore * torch.cat([I_high,I_high,I_high],dim=1)

            L_output_np = L_output.detach().cpu().numpy()[0]
            R_restore_np = R_restore.detach().cpu().numpy()[0]
            # I_low_np = I_low.detach().cpu().numpy()[0]
            R_low_np = R_low.detach().cpu().numpy()[0]
            R_high_np = R_high.detach().cpu().numpy()[0]
            L_low_np = L_low_tensor.numpy()[0]
            L_high_np = L_high_tensor.numpy()[0]

            sample_imgs = np.concatenate((R_low_np, R_restore_np, R_high_np, L_low_np, L_output_np, L_high_np), axis=0 )
            
            filepath = os.path.join(plot_dir, f'{name[0]}_epoch_{epoch//100}.png')
            split_point = None#[0, 3, 6, 9, 10, 13, 16]
            img_dim = L_high_np.shape[1:]
            sample(sample_imgs, split=split_point, figure_size=(2, 3), 
                        img_dim=img_dim, path=filepath, num=epoch, metrics=True)

if __name__ == "__main__":
    criterion = Restore_Loss()
    model = RestoreNet_Unet(use_MaskMul=False)
    decom_net = DecomNet()

    parser = BaseParser()
    args = parser.parse()

    with open(args.config) as f:
        config = yaml.load(f)
    args.checkpoint = True

    if args.checkpoint is not None:
        decom_net = load_weights(decom_net, path='./weights/decom_net_normal.pth')
        log('DecomNet loaded from decom_net.pth')
        model = load_weights(model, path='./weights/restore_net_finetune.pth')# restore-SID/restore_mask_0.pth')
        log('Model loaded from restore_net.pth')

    root_path_train = r'H:\datasets\Low-Light Dataset\KinD++\LOLdataset\our485'
    root_path_test = r'C:\DeepLearning\KinD_plus-master\LOLdataset\eval15'
    list_path_train = build_LOLDataset_list_txt(root_path_train)
    list_path_test = build_LOLDataset_list_txt(root_path_test)
    # list_path_train = os.path.join(root_path_train, 'pair_list.csv')
    # list_path_test = os.path.join(root_path_test, 'pair_list.csv')

    log("Buliding LOL Dataset...")
    dst_train = LOLDataset(root_path_train, list_path_train,
                            crop_size=config['length'], to_RAM=True)
    dst_test = LOLDataset(root_path_test, list_path_test,
                            crop_size=config['length'], to_RAM=True, training=False)

    train_loader = DataLoader(dst_train, batch_size = config['batch_size'], shuffle=True)
    # train_loader = data_prefetcher(train_loader)
    test_loader = DataLoader(dst_test, batch_size=1)

    trainer = Restore_Trainer(config, train_loader, criterion, model, 
                            dataloader_test=test_loader, decom_net=decom_net)

    if args.mode == 'train':
        trainer.train()
    else:
        trainer.test(plot_dir='./images/samples-restore')