# SuperQuality
 Improve Image Quality

本科毕设，基本架构类似我之前的复现[KinD-Pytorch](https://github.com/fenghansen/KinD-pytorch)，改进在以下方面：
1. 分解网络我记得和KinD是一样的
2. 恢复网络我魔改了，把EnlightenGAN的mask、channel attention、GAN都引入进来了
3. 提亮网络我设计了个快速的传统算法版
4. 额外增加了用VDN前后处理去噪的选项

这个库可以运行``SuperQualityGUI.py``在图形化界面中调试，可以保存全部中间结果：

![newspaper_extra](https://user-images.githubusercontent.com/39181837/206389803-05021717-8e06-48a9-92f9-a6aaaf07c3cf.png)

KinD-pytorch我就不维护了，当时照抄那个loss如何也调试不出来，官方模型也没调通，复现得脑壳痛。不过后来有师弟复现出来过，很神奇，可能是版本问题吧。
