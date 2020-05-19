import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import time
import yaml
import sys
from torchvision import transforms
from torchsummary import summary
from base_trainer import BaseTrainer
from base_parser import BaseParser
from losses import *
from models import *
from dataloader import *

class Denoise_Trainer(BaseTrainer):
    def train(self):
        print(f'Using device {self.device}')
        self.model.to(device=self.device)
        summary(self.model, input_size=(3, 128, 128))
        # faster convolutions, but more memory
        # cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9984)
        try:
            for iter in range(self.epochs):
                idx = 0
                iter_start_time = time.time()
                gpu_time = 0

                for low_tensor, high_tensor, name in self.dataloader:
                    low = low_tensor.to(self.device)
                    high = high_tensor.to(self.device)
                    gpu_time_iter = time.time()
                    output = self.model(low)
                    loss = self.loss_fn(output, high)
                    if idx % 10 == 0:
                        print(f"iter: {iter}_{idx}\taverage_loss: {loss.item():.6f}")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    idx += 1
                    gpu_time += time.time()-gpu_time_iter 

                if iter % self.print_frequency == 0:
                    self.test(iter, plot_dir='./images/samples-denoise')

                if iter % self.save_frequency == 0:
                    torch.save(self.model.state_dict(), f'./weights/denoise/denoise_{iter//100}.pth')
                    log("Weight Has saved as 'decom_net.pth'")
                        
                scheduler.step()
                iter_end_time = time.time()
                log(f"Time taken: {iter_end_time - iter_start_time:.3f} sec, include gpu_time {gpu_time:.3f} sec\t lr={scheduler.get_lr()[0]:.6f}")

        except KeyboardInterrupt:
            torch.save(self.model.state_dict(), 'INTERRUPTED_denoise.pth')
            log('Saved interrupt_decom')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

    @no_grad
    def test(self, epoch=-1, plot_dir='./images/samples-denoise'):
        self.model.eval()
        for low_tensor, high_tensor, name in self.dataloader_test:
            low = low_tensor.to(self.device)
            high = high_tensor.to(self.device)
            with torch.no_grad():
                output = self.model(low)

            output_np = output.detach().cpu().numpy()[0]
            low_np = low_tensor.numpy()[0]
            high_np = high_tensor.numpy()[0]
            sample_imgs = np.concatenate( (low_np, output_np, high_np), axis=0 )
            filepath = os.path.join(plot_dir, f'{name[0]}_epoch_{epoch//100}.png')
            split_point = [0, 3, 6, 9]
            img_dim = low_np.shape[1:]
            sample(sample_imgs, split=split_point, figure_size=(1, 3), 
                        img_dim=img_dim, path=filepath, num=epoch, metrics=True)


if __name__ == "__main__":
    criterion = Decom_Loss()
    model = DecomNet()

    parser = BaseParser()
    args = parser.parse()
    # args.checkpoint = True
    # if args.checkpoint is not None:
    #     model = load_weights(model, path='./weights/decom-SID/decom_net_1.pth')
    #     print('Model loaded from decom_net.pth')

    with open(args.config) as f:
        config = yaml.load(f)

    root_path = r'H:\datasets\SIDD\SIDD_Medium_Srgb'

    log("Buliding SIDD Dataset...")
    # transform = transforms.Compose([transforms.ToTensor(),])
    config['length'] = 64
    dst_train = SIDD_Dataset(root_path, crop_size=config['length'], to_RAM=False, crops_per_image=8)
    dst_test = SIDD_Dataset(root_path, crop_size=config['length'], to_RAM=True, training=False)

    train_loader = DataLoader(dst_train, batch_size = config['batch_size'], shuffle=True)
    test_loader = DataLoader(dst_test, batch_size=1)
    train_loader = crops_reshape(train_loader)
    test_loader = crops_reshape(test_loader)

    trainer = Denoise_Trainer(config, train_loader, criterion, model, dataloader_test=test_loader)
    # --config ./config/config.yaml
    if args.mode == 'train':
        trainer.train()
    else:
        trainer.test(plot_dir='./images/samples-denoise')