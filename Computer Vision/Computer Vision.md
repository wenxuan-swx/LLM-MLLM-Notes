# 图像的基本表示

图像的张量表示: (高, 宽, 通道)

通道可以为1, 3, 4(2一般是灰度通道, 3一般是RGB或者HSV, 4一般是CMYK或者RGBA) , 每个通道都有一个高*宽的矩阵组成, 每个像素点上的值在0-255之间, 表示对应的灰度
 
*注意opencv在读取图像时默认的通道顺序是BGR*, 需要进行手动转换
```python
img = cv2.cvtColor(img, 
                   cv2.COLOR_BGR2RGB)  # 转换方式
```
一些基本理解:
1.   给图像加上某个数值, 就会让图像变亮
2.   给图像减去某个数值, 就会让图像变暗
3.   给图像乘上某个数值, 就会让图像变鲜艳(调整对比度)

# 卷积和卷积神经网络

## 卷积
二维矩阵的卷积表示其中一个矩阵在平面上旋转180度后, 与另一个矩阵求**点积**的结果(卷 = 旋转, 积 = 点积)

但是实际应用中, 我们不再进行旋转这一步, 直接进行点积操作

我们把旋转后的矩阵的值称为权重, 把该矩阵称为**过滤器**, 也叫做**卷积核**

由于图像像素矩阵往往很大, 我们不可能创造一个和原图像素矩阵一样大的权重矩阵来进行卷积运算. 因此我们往往创造一个较小的矩阵, 然后对像素矩阵进行扫描: 
![alt text](image.png)
![alt text](image-1.png)
![alt text](image-2.png)

每个卷积核在原图上扫描的区域(即绿色区域)被称为**感受野**, 卷积核与感受野轮流点积得到的新矩阵被叫做**特征图**

使用不同的卷积核, 可以达到不同的效果, 比如边缘检测(索贝尔算子、拉普拉斯算子), 锐化, 模糊等

## 卷积神经网络
如何寻找效果更好的卷积核? 

深度学习的核心思想之一, 就是给算法训练目标, 让算法自己找最佳参数. 于是卷积神经网络就诞生了

### 参数共享
由于参数共享的性质, 卷积神经网络可以节省很大的参数量
*   **全连接DNN**: 假设图片尺寸为(600, 400), 输入DNN时需要将像素拉平至一维, 在输入层上就需要占用400*600=24w个神经元; 假设若干个隐藏层的神经元数为1w个, 则总共需要24亿个参数才能解决问题
*   **卷积神经网络CNN**: 无论图片多大, **卷积神经网络要求解的参数就是卷积核上的所有数字**. 假设卷积核的尺寸是5*5, 隐藏层的神经元个数为1w, 那么只需要求解25w个参数
  

### 稀疏交互
特征图上的一个神经元, 只和对应的感受野上极少数神经元有关. 因此, 神经元之间并不是全连接, 而是稀疏连接. 一般认为, 稀疏交互让CNN获得了提取更深特征的能力, 因为:
*   更符合数据的局部相关性假设
*   降低了过拟合风险

# 在pytorch中构筑卷积神经网络

## 卷积层
处理图像的普通卷积nn.Conv2d:

`torch.nn.Conv2d`(*in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
groups=1, bias=True, padding_mode='zeros'*)

*   **kernel_size**: 对于经典架构, 论文中所使用的尺寸就是最好的尺寸; 如果是自己的神经网络, 3*3几乎就是最好的选择
    *   卷积核几乎都是正方形
    *   卷积核的尺寸最好是奇数
    *   卷积核的尺寸往往比较小
*   **in_channels, out_channels**: in_channels输入卷积层的图像的通道数, out_channels是输出的特征图的数量
    *   卷积层的输入时图像时: 在一次扫描中, 假设输入了3通道的图片, 那么会有3个尺寸相同但数值不同的卷积核对3个通道分别进行扫描, 然后将得到的结果相加, 形成一张新的特征图. 因此一次扫描对应一张特征图, **输出特征图的个数实际上就是扫描的次数** (对同一个通道提供不同的卷积核来进行多次扫描是比较常见的操作)
    *   在这个过程中, 所需要的卷积核的数量 = in_channels * out_channels
    *   卷积层的输入是上层特征图时: **特征图会被当作通道对待, 一次扫描会扫描所有的特征图, 然后加和成一个新的特征图**. 在这个过程中, 也是一次扫描对应一个特征图
    *   无论在哪一层, 生成的特征图的数量都等于这一层的扫描次数, 也就是等于out_channels. **下一层卷积的in_channels就等于上一层卷积的out_channels**
*   **bias**: 可以理解为乘完权重之后加上的常数项. 在扫描完生成特征图之后加入特征图矩阵的每个元素中
*   **stride**: 卷积操作中的步长, 即每次卷积操作完之后向右移动或每次扫描完一整行后向下移动的像素数
    *   当步长=1时, 特征图的尺寸在扫描过后会发生改变. 假设卷积核的尺寸是$(K_H, K_W)$, 特征图的尺寸为$(H_{out}, W_{out})$, 那么有:
       $$H_{out} = H_{in} - K_H +1$$
       $$W_{out} = W_{in} - K_W +1$$
    *   设置步长后, 可以对生成的特征图进行降维(减少像素量), 设横向步长和纵向步长分别S[0]和S[1], 于是有:
       $$H_{out} = \frac{H_{in} - K_H}{S[0]} +1$$
       $$W_{out} = \frac{W_{in} - K_W}{S[1]} +1$$
*   **padding**: 四周填充的像素数(上下左右都会填充, 所以通道实际尺寸会增加2*padding)
     $$H_{out} = \frac{H_{in} - K_H +2*P}{S[0]} +1$$
       $$W_{out} = \frac{W_{in} - K_W+2*P}{S[1]} +1$$
    *   当有步长的时候, 往往容易出现扫描不充分/不完全的情况
    *   如果步长小于卷积核, 那么中间的区域就会反复被扫描, 但边缘的像素只会被捕捉到一两次, 导致提取信息不均衡, 中间重边缘轻
    *   在pytoch中, 一旦出现小数, 就向下取整, 因为没办法被扫描完全的部分会直接被丢弃

## 池化层

池化是一种简单粗暴的降维方式, 常见的有平均池化(`torch.nn.AvgPool2d`)和最大池化(`torch.nn.MaxPool2d`)
![alt text](image-4.png)
![alt text](image-3.png)

池化层也有核, 但只有尺寸, 没有值

默认步长 = 核尺寸, 这样可以避免重复扫描

池化后输出的特征图尺寸仍然满足:
$$H_{out} = \frac{H_{in} - K_H +2*P}{S[0]} +1$$
$$W_{out} = \frac{W_{in} - K_W +2*P}{S[1]} +1$$

比如, 对于(2, 2)的池化层来说, 可以使行列数量都减半, 也就是使总像素数量减少3/4, 是非常有效的降维方式

由于我们只关心池化后的输出的图的尺寸, 因此也可以使用自适应池化层, 只需要输入output_size即可

`torch.nn.AdaptiveMaxPool2d(output_size)`
`torch.nn.AdaptiveAvgPool2d(output_size)`

池化层的其他特点:
1.   提供非线性变化. 卷积层本质还是线性变化, 所以通常会在卷积层的后面增加激活层, 使用非线性激活函数
2.   有一定的平移不变性
3.   池化层的所有参数都是超参数, 没有任何可以学习的参数, 这既是优点(增加池化层不会增加参数量), 也是致命问题(池化层不会随着算法一起进步)
4.   对不同特征的特征图安相同的规律进行一次性降维, 必然引起大规模的信息损失

## BatchNorm2d, Droupout
`nn.BatchNorm2d`*(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)*

**BatchNorm2d**起到的是归一化的作用. 当出现在卷积网络前后时, BatchNorm2d所需要输入的是上一层输出的特征图的数量


`torch.nn.Droupout2d`*(p=0.5, inplace=False)*

**Droupout**指的是在训练过程中, 以概率p随机沉默一部分神经元, 以此减少过拟合可能性. 在卷积中, **Droupout不会以神经元为单位, 而是以特征图为单位进行沉默**, 所以不要使用太大的p,否则容易陷入欠拟合.

这两个层在训练和测试的时候起作用的方式不太一样, 使用时一定要区分训练和测试的情况

# LeNet5
![alt text](image-6.png)

*   共6个层: 2个卷积层+2个平均池化层+2个全连接层
*   每个卷积层和全连接层后都适用激活函数tanh或sigmoid
*   **输入->(卷积+池化)->(卷积+池化)->(线性x2)->输出**

```python
import torch
from torch import nn
from torch.nn import functional as F

data = torch.ones(size=(10,1,32,32))  #(sample_num, 通道, 高, 宽)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.pool1 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.pool2 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(16*5*5,120) #这里的上层输入是图像中的全部像素 self.fc2 = nn.Linear(120,84)

    def forward(self,x):
        x = F.tanh(self.conv1(x))
        x = self.pool1(x)
        x = F.tanh(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1,16*5*5) #需要将数据的特征部分“拉平”才能够进入FC层 
        x = F.tanh(self.fc1(x))
        output = F.softmax(self.fc2(x),dim=1)  #(sample, feature)

net = Model()

net(data)


#卷积网络的一大特点是:改变输入数据的尺寸，整个网络就会开始花式报错 
#你可以试试将数据32*32的结构修改至28*28
#安装torchinfo库，查看网络结构 

!pip install torchinfo
from torchinfo import summary
net = Model()
summary(net, input_size=(10, 1, 32, 32))
```

# AlexNet
![alt text](image-7.png)

*   共11层: 5个卷积层, 3个池化层, 1个隐藏全连接层, 1个全连接输出层
*   在全连接层的前后使用了Dropout层(图中未表示)
*   **输入->(卷积+池化)->(卷积+池化)->(卷积x3+池化)->(线性x3)->输出**

与LeNet5相比, AlexNet的主要改进有:
1.   **卷积核更小, 网络更深, 通道数更多**
     *   小卷积核一定要搭配大通道来使用, 才能尽可能多地提取信息
     *   小卷积核会使得特征图尺寸下降更慢, 从而允许更深的网络. 图像天生就适合用更深的网络来提取信息
2.   使用ReLU激活函数

| 激活函数  | 输出范围   | 具体函数                                | 是否有梯度消失问题       | 是否以 0 为中心 | 计算效率 | 常见用途                                     |
|-----------|------------|-----------------------------------------|--------------------------|----------------|----------|---------------------------------------------|
| **ReLU**  | \[0, +∞\]  | $f(x) = \max(0, x)$                  | 无（但有死神经元问题）   | 否             | 高       | 深度神经网络的默认激活函数                    |
| **Sigmoid** | \[0, 1\]   | $f(x) = \frac{1}{1 + e^{-x}}$        | 有                       | 否             | 低       | 二分类问题，早期神经网络                     |
| **Tanh**  | \[-1, 1\]  | $f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | 有 | 是 | 中 | 需要平滑过渡且对负值敏感的场景 |
3.   使用Dropout层用来控制过拟合
4.   引入各种图像增强技术来扩大数据集, 进一步控制过拟合
5.   使用GPU来进行训练
   
```python

import torch
from torch import nn
from torch.nn import functional as F
data = torch.ones(size=(10,3,227,227)) #假设图像的尺寸为227x227
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        #大卷积核、较大的步长, 快速降低图片尺寸; 较多的通道弥补降低图片尺寸的信息损失
        self.conv1 = nn.Conv2d(1,96,kernel_size=11, stride=4) 
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)

        #卷积核、步长恢复正常大小，进一步扩大通道
        self.conv2 = nn.Conv2d(96,256,kernel_size=5, padding=2)    #增加padding, 不让图片尺寸缩小
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2)

        #连续的卷积层，疯狂提取特征
        self.conv3 = nn.Conv2d(256,384,kernel_size=3,padding=1) 
        self.conv4 = nn.Conv2d(384,384,kernel_size=3,padding=1) 
        self.conv5 = nn.Conv2d(384,256,kernel_size=3,padding=1) 
        self.pool3 = nn.MaxPool2d(kernel_size=3,stride=2)

        #全连接层
        self.fc1 = nn.Linear(256*6*6,4096) #这里的上层输入是图像中的全部像素 
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,1000) #输出ImageNet的一千个类别

   def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)

        x = x.view(-1,256*6*6) #需要将数据的特征部分“拉平”才能够进入FC层
        x = F.relu(F.dropout(self.fc1(x),0.5)) #dropout:随机让50%的权重为0 
        x = F.relu(F.dropout(self.fc2(x),0.5))
        output = F.softmax(self.fc3(x),dim=1)

net = Model()
net(data)
```

# 自己搭建卷积神经网络
模型评估三角: **效果, 效率, 可解释性**

“**小卷积核、多通道、更深层**”被证明是有效的

## 网络的深度
更深的网络会展现出更强大的学习能力

但可能碰到的问题是:
1.   输入图像的尺寸会限制深度的选择
     *   **卷积层和池化层都能快速地降低特征图的尺寸**. 以224x224尺寸为例, 只要池化层和步长为2的卷积层出现5次, 特征图的尺寸就会变成7x7, 不再具备追求更深网络的空间
     *   我们只能依赖卷积层来控制特征图尺寸缩小的速度. 卷积核越大, 生成的特征图尺寸越小. **因此为了深度, 卷积核最好选择小尺寸. 但对于小卷积核, padding就无法被设置得太大(padding参数要小于核尺寸的一半)**
     *   为了增加深度, 常见处理思路有:
         *   **递减架构**: **不使用池化层**, 利用padding核卷积核搭配, 使每经过一次卷积层, 就缩小几个像素
         *   **重复架构(计算效率更高、鲁棒性更好)**: 利用padding核卷积核搭配, 使**每次经过卷积层的尺寸不变**, 利用池化层来缩小特征图

### VGGNet
基于重复架构, VGGNet的核心思想是**使用多个连续且保持特征图尺寸不变的卷积层来增加深度**, 以增加算法的学习能力

![alt text](image-8.png)

*   n x 卷积+池化 = 1block