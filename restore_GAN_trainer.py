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

class Restore_GAN_Trainer(BaseTrainer):
    def __init__(self, config, dataloader, criterion, model, gan_mode='RaSGAN',
            dataloader_test=None, decom_net=None, discriminator=None):
        super().__init__(config, dataloader, criterion, model, dataloader_test)
        self.gan_mode = gan_mode
        self.D = discriminator
        self.D.to(device=self.device)
        self.decom_net = decom_net
        self.decom_net.to(device=self.device)
        # EMA初始化
        self.ema_D = EMA(self.D, 0.9)
        self.ema_G = EMA(self.model, 0.9)
        self.ema_D.register()
        self.ema_G.register()

    def train(self):
        # print(self.model)
        summary(self.model, input_size=[(3, 256, 256), (1,256,256)])

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9984) #0.977237, 0.986233
        scheduler_D = lr_scheduler.ExponentialLR(optimizer_D, gamma=0.9984) #0.977237, 0.986233
        try:
            for iter in range(self.epochs):
                idx = 0
                iter_start_time = time.time()
                gpu_time = 0
                loss_frequency = 6
                loss_average_D = 0
                loss_average_G = 0

                for L_low_tensor, L_high_tensor, name in self.dataloader:
                    optimizer.zero_grad()
                    optimizer_D.zero_grad()
                    L_low = L_low_tensor.to(self.device)
                    L_high = L_high_tensor.to(self.device)

                    gpu_time_iter = time.time()
                    with torch.no_grad():
                        R_low, I_low = self.decom_net(L_low)
                        R_high, I_high = self.decom_net(L_high)
                        
                    # R_restore = self.model(R_low, I_low)
                    R_restore,R2,R4,R8 = self.model(R_low, I_low, mode='train')
                    # reconstract loss
                    recon_loss = self.loss_fn(R_restore, R_high,R2,R4,R8)
                    # GAN loss
                    G_fake = R_restore.detach()
                    G2 = R2.detach(); G4 = R4.detach(); G8 = R8.detach()
                    ref2, ref4, ref8 = Pyramid_Sample(R_high)
                    D_real = self.D(R_high,ref2,ref4,ref8)
                    D_fake = self.D(G_fake,G2,G4,G8)
                    D_fake_for_G = self.D(R_restore,R2,R4,R8)
                    y_ones = torch.ones_like(D_real).to(self.device)
                    y_zeros = torch.zeros_like(D_fake).to(self.device)
                    
                    if self.gan_mode == 'RSGAN':
                        ### Relativistic Standard GAN
                        BCE_stable = torch.nn.BCEWithLogitsLoss()
                        # Discriminator loss
                        errD = BCE_stable(D_real - D_fake, y_ones)
                        loss_D = torch.mean(errD)
                        # Generator loss
                        errG = BCE_stable(D_fake_for_G - D_real, y_ones)
                        loss_G = torch.mean(errG)
                    elif self.gan_mode == 'SGAN':
                        criterion = torch.nn.BCEWithLogitsLoss()
                        # Real data Discriminator loss
                        errD_real = criterion(D_real, y_ones)
                        # Fake data Discriminator loss
                        errD_fake = criterion(D_fake, y_zeros)
                        loss_D = torch.mean(errD_real + errD_fake) / 2
                        # Generator loss
                        errG = criterion(D_fake_for_G, y_ones)
                        loss_G = torch.mean(errG)
                    elif self.gan_mode == 'RaSGAN':
                        BCE_stable = torch.nn.BCEWithLogitsLoss()
                        # Discriminator loss
                        errD = (BCE_stable(D_real - torch.mean(D_fake), y_ones) + 
                                BCE_stable(D_fake - torch.mean(D_real), y_zeros))/2
                        loss_D = torch.mean(errD)
                        # Generator loss
                        errG = (BCE_stable(D_real - torch.mean(D_fake), y_zeros) + 
                                BCE_stable(D_fake - torch.mean(D_real), y_ones))/2
                        loss_G = torch.mean(errG)

                    loss_average_D += loss_D.item()
                    loss_G += recon_loss * 0.01
                    loss_average_G += loss_G.item()

                    if idx % loss_frequency == loss_frequency-1:
                        loss_average_G /= loss_frequency
                        loss_average_D /= loss_frequency
                        log(f"iter: {iter}_{idx}\tloss_G: {loss_average_D:.6f} - loss_D: {loss_average_G:.6f}")
                        loss_average_D = 0
                        loss_average_G = 0
                    
                    # backward, "retain_graph=True" because G need grad from D_real
                    loss_D.backward(retain_graph=True)
                    optimizer_D.step()
                    loss_G.backward()
                    optimizer.step()
                    # 训练过程中，更新完参数后，同步update shadow weights
                    self.ema_D.update()
                    self.ema_G.update()
                    gpu_time += time.time()-gpu_time_iter 
                    idx += 1

                if iter % self.print_frequency == 0:
                    self.test(iter, plot_dir='./images/samples-restore-gan-pyramid')

                if iter % self.save_frequency == 0:
                    torch.save(self.model.state_dict(), f'./weights/restore-GAN-pyramid/restore_GAN_{iter//100}.pth')
                    torch.save(self.D.state_dict(), f'./weights/restore-GAN-pyramid/D_pyramid_{iter//100}.pth')
                    log("Weight Has saved as 'restore_GAN.pth'")
                
                scheduler.step()
                scheduler_D.step()
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
        # eval前，apply shadow weights
        self.ema_G.apply_shadow()
        for L_low_tensor, L_high_tensor, name in self.dataloader_test:
            L_low = L_low_tensor.to(self.device)
            L_high = L_high_tensor.to(self.device)

            R_low, I_low = self.decom_net(L_low)
            R_high, I_high = self.decom_net(L_high)

            R_restore = self.model(R_low, I_low)
            L_output = R_restore * torch.cat([I_high,I_high,I_high],dim=1)

            L_output_np = L_output.detach().cpu().numpy()[0]
            R_restore_np = R_restore.detach().cpu().numpy()[0]
            I_low_np = I_low.detach().cpu().numpy()[0]
            R_low_np = R_low.detach().cpu().numpy()[0]
            R_high_np = R_high.detach().cpu().numpy()[0]
            L_high_np = L_high_tensor.numpy()[0]

            sample_imgs = np.concatenate((R_low_np, R_restore_np, R_high_np, 1-I_low_np, L_output_np, L_high_np), axis=0 )
            
            filepath = os.path.join(plot_dir, f'{name[0]}_epoch_{epoch//100}.png')
            split_point = [0, 3, 6, 9, 10, 13, 16]
            img_dim = L_high_np.shape[1:]
            sample(sample_imgs, split=split_point, figure_size=(2, 3), 
                        img_dim=img_dim, path=filepath, num=epoch)
        # eval之后，恢复原来模型的参数
        self.ema_G.restore()

if __name__ == "__main__":
    criterion = Restore_Loss()
    model = RestoreNet_Unet()
    decom_net = DecomNet()
    discriminator = Restore_D_Pyramid()

    parser = BaseParser()
    args = parser.parse()

    with open(args.config) as f:
        config = yaml.load(f)
    args.checkpoint = True

    if args.checkpoint is not None:
        discriminator = load_weights(discriminator, path='./weights/restore-GAN-pyramid/D_single_1.pth')
        log('Discriminator loaded from D_pyramid.pth')
        decom_net = load_weights(decom_net, path='./weights/decom_net_normal.pth')
        log('DecomNet loaded from decom_net.pth')
        model = load_weights(model, path='./weights/restore-GAN-pyramid/restore_GAN_1.pth')
        log('Model loaded from restore_net.pth')

    root_path_train = r'H:\datasets\Low-Light Dataset\KinD++\LOLdataset\our485'
    root_path_test = r'C:\DeepLearning\KinD_plus-master\LOLdataset\eval15'
    list_path_train = build_LOLDataset_list_txt(root_path_train)
    list_path_test = build_LOLDataset_list_txt(root_path_test)
    # list_path_train = os.path.join(root_path_train, 'pair_list.csv')
    # list_path_test = os.path.join(root_path_test, 'pair_list.csv')

    log("Buliding LOL Dataset...")
    # transform = transforms.Compose([transforms.ToTensor()])
    dst_train = LOLDataset(root_path_train, list_path_train,
                            crop_size=config['length'], to_RAM=True)
    dst_test = LOLDataset(root_path_test, list_path_test,
                            crop_size=config['length'], to_RAM=True, training=False)

    train_loader = DataLoader(dst_train, batch_size = config['batch_size'], shuffle=True)
    # train_loader = data_prefetcher(train_loader)
    test_loader = DataLoader(dst_test, batch_size=1)

    trainer = Restore_GAN_Trainer(config, train_loader, criterion, model, dataloader_test=test_loader,
                            discriminator=discriminator, decom_net=decom_net)

    if args.mode == 'train':
        trainer.train()
    else:
        trainer.test()