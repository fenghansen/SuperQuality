import torch
import pytorch_colors as colors
from dataloader import *

grads = {}

def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

x = torch.randn((2,2), requires_grad=True)
# print(x)
# print(x>0.5)
# print((x>0.2) & (x<0.7))
y = 3*x
z = y * y

L_high = cv2.imread(r'C:\DeepLearning\KinD_plus-master\LOLdataset\eval15\high\780.png')
# cv2.imshow('high', L_high)
L_high = np.array(L_high).astype(np.float32).transpose(2,0,1) / 255. #[np.newaxis,:,:,:]
cv2.imshow('high', L_high.transpose(1,2,0))

L_high_tensor = torch.as_tensor(L_high)
L_h = L_high_tensor
L_hsv = rgb2hsv(L_h, dim=0)
# L_color_hsv = colors.rgb_to_hsv(L_h)
# L_hsv[-1] = torch.ones(400,600)
L_hsv = L_hsv.numpy().transpose(1,2,0)
cv2.imshow('hsv1', L_hsv)
# L_color_hsv = L_color_hsv.numpy().transpose(1,2,0)
# cv2.imshow('hsv2', L_color_hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()

L_high_rgb = hsv2rgb(L_high_hsv)

L_rgb = L_high_rgb.numpy().transpose(1,2,0)
cv2.imshow('rgb', L_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

L = L_high_hsv * 255.
loss_mean = torch.mean(L)
loss = loss_mean

# # 为中间变量注册梯度保存接口，存储梯度时名字为 y。
# y.register_hook(save_grad('y'))

# # # 反向传播 
# # z.backward()


# # 为中间变量注册梯度保存接口，存储梯度时名字为 L。
# loss_mean.register_hook(save_grad('L'))

# # 反向传播 
# loss.backward()

# # 查看 y 的梯度值
# print(grads['y'])
# # 查看 L 的梯度值
# print(f"L:{grads['L']}")
# print(L)