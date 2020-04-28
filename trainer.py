from torch.optim import lr_scheduler
import yaml
import sys
from torchvision import transforms
from torchsummary import summary
from base_trainer import BaseTrainer
from losses import *
from models import *
from base_parser import BaseParser
from dataloader import *

class Trainer(BaseTrainer):
    def train(self):
        print(f'Using device {self.device}')
        self.model.to(device=self.device)
        summary(self.model, input_size=(3, 128, 128))
        # faster convolutions, but more memory
        # cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.997)
        try:
            for iter in range(self.epochs):
                epoch_loss = 0
                idx = 0
                iter_start_time = time.time()
                for L_low_tensor, L_high_tensor, name in self.dataloader:
                    # Color Space Conversion
                    L_low_hsv = colors.rgb_to_hsv(L_low_tensor)
                    L_high_hsv = colors.rgb_to_hsv(L_high_tensor)
                    # Send tensor to CUDA 
                    L_low = L_low_hsv.to(self.device)
                    L_high = L_high_hsv.to(self.device)
                    # Foward Denoise
                    L_recon = self.model(L_low)
                    # Compute Loss
                    loss = self.loss_fn(L_recon, L_high)
                    if idx % 8 == 0:
                        print(f"iter: {iter}_{idx}\taverage_loss: {loss.item():.6f}")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    idx += 1

                if iter % self.print_frequency == 0:
                    self.test(iter, plot_dir='./images/samples-unetonly')

                if iter % self.save_frequency == 0:
                    torch.save(self.model.state_dict(), './weights/unetonly.pth')
                    log("Weight Has saved as 'unetonly.pth'")
                        
                scheduler.step()
                iter_end_time = time.time()
                log(f"Time taken: {iter_end_time - iter_start_time:.3f} seconds\t lr={scheduler.get_lr()[0]:.6f}")

        except KeyboardInterrupt:
            torch.save(self.model.state_dict(), './weights/INTERRUPTED_unetonly.pth')
            log('Saved INTERRUPTED_unetonly')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

    @no_grad
    def test(self, epoch=-1, plot_dir='./images/samples-unetonly'):
        self.model.eval()
        hook = 0
        with torch.no_grad():
            for L_low_tensor, L_high_tensor, name in self.dataloader_test:
                # Color Space Conversion
                L_low_hsv = colors.rgb_to_hsv(L_low_tensor)
                L_high_hsv = colors.rgb_to_hsv(L_high_tensor)
                # Send tensor to CUDA 
                L_low = L_low_hsv.to(self.device)
                L_high = L_high_hsv.to(self.device)
                # Foward Denoise
                L_recon = self.model(L_low)
                L_output = colors.hsv_to_rgb(L_recon)

                L_output_np = L_output.detach().cpu().numpy()[0]
                L_low_np = L_low_tensor.numpy()[0]
                L_high_np = L_high_tensor.numpy()[0]
                sample_imgs = np.concatenate( (L_low_np, L_output_np, L_high_np), axis=0 )
                filepath = os.path.join(plot_dir, f'{name[0]}_epoch_{epoch}.png')
                split_point = [0, 3, 6, 9]
                img_dim = L_low_np.shape[1:]
                sample(sample_imgs, split=split_point, figure_size=(1, 3), 
                            img_dim=img_dim, path=filepath, num=epoch)


if __name__ == "__main__":
    criterion = Unet_Loss()
    model = Unet()

    parser = BaseParser()
    args = parser.parse()
    # args.checkpoint = True
    # if args.checkpoint is not None:
    #     pretrain = torch.load('./weights/unetonly.pth')
    #     model.load_state_dict(pretrain)
    #     print('Model loaded from unetonly.pth')

    with open(args.config) as f:
        config = yaml.load(f)

    root_path_train = r'H:\datasets\Low-Light Dataset\KinD++\LOLdataset\our485'
    root_path_test = r'C:\DeepLearning\KinD_plus-master\LOLdataset\eval15'
    list_path_train = build_LOLDataset_list_txt(root_path_train)
    list_path_test = build_LOLDataset_list_txt(root_path_test)
    # list_path_train = os.path.join(root_path_train, 'pair_list.csv')
    # list_path_test = os.path.join(root_path_test, 'pair_list.csv')

    log("Buliding LOL Dataset...")
    # transform = transforms.Compose([transforms.ToTensor(),])
    dst_train = LOLDataset(root_path_train, list_path_train,
                            crop_size=config['length'], to_RAM=True)
    dst_test = LOLDataset(root_path_test, list_path_test,
                            crop_size=config['length'], to_RAM=True, training=False)

    train_loader = DataLoader(dst_train, batch_size = config['batch_size'], shuffle=True)
    test_loader = DataLoader(dst_test, batch_size=1)

    trainer = Trainer(config, train_loader, criterion, model, dataloader_test=test_loader)
    # --config ./config/config.yaml
    if args.mode == 'train':
        trainer.train()
    else:
        trainer.test()