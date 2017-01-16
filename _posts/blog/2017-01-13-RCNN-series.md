---
layout: post
title: RCNN,Fast RCNN,Faster RCNN 总结
description: 深度学习
category: blog
---

## 一 背景知识

### 1.1  IOU的定义

物体检测需要定位出物体的bounding box，就像下面的图片一样，我们不仅要定位出车辆的bounding box 我们还要识别出bounding box 里面的物体就是车辆。对于bounding box的定位精度，有一个很重要的概念，因为我们算法不可能百分百跟人工标注的数据完全匹配，因此就存在一个定位精度评价公式：IOU。

![iou1](/images/blog/rcnn1.jpg)

IOU定义了两个bounding box的重叠度，如下图所示:

![iou1](/images/blog/rcnn2.jpg)

矩形框A、B的一个重合度IOU计算公式为：

IOU=(A∩B)/(A∪B)

就是矩形框A、B的重叠面积占A、B并集的面积比例:

IOU=SI/(SA+SB-SI)

### 1.2 非极大值抑制

 RCNN算法，会从一张图片中找出n多个可能是物体的矩形框，然后为每个矩形框为做类别分类概率：

![iou1](/images/blog/rcnn3.jpg)

就像上面的图片一样，定位一个车辆，最后算法就找出了一堆的方框，我们需要判别哪些矩形框是没用的。非极大值抑制：先假设有6个矩形框，根据分类器类别分类概率做排序，从小到大分别属于车辆的概率分别为A、B、C、D、E、F。

1. 从最大概率矩形框F开始，分别判断A~E与F的重叠度IOU是否大于某个设定的阈值;

2. 假设B、D与F的重叠度超过阈值，那么就扔掉B、D；并标记第一个矩形框F，是我们保留下来的。

3. 从剩下的矩形框A、C、E中，选择概率最大的E，然后判断E与A、C的重叠度，重叠度大于一定的阈值，那么就扔掉；并标记E是我们保留下来的第二个矩形框。

就这样一直重复，找到所有被保留下来的矩形框。

### 1.3 一张图概览RCNN

![RCNN相关方法对比](/images/blog/rcnn11.png)

## 二 RCNN

 算法概要：首先输入一张图片，我们先定位出2000个物体候选框，然后采用CNN提取每个候选框中图片的特征向量，特征向量的维度为4096维，接着采用svm算法对各个候选框中的物体进行分类识别。也就是总个过程分为三个程序：a、找出候选框；b、利用CNN提取特征向量；c、利用SVM进行特征向量分类。具体的流程如下图片所示：

![iou1](/images/blog/rcnn4.png)

下面分别讲解各个步骤。

### 2.1 候选框搜索

 当我们输入一张图片时，我们要搜索出所有可能是物体的区域，这个采用的方法是传统文献的算法selective search (github上有源码)，通过这个算法我们搜索出2000个候选框。然后从上面的总流程图中可以看到，搜出的候选框是矩形的，而且是大小各不相同。然而CNN对输入图片的大小是有固定的，如果把搜索到的矩形选框不做处理，就扔进CNN中，肯定不行。因此对于每个输入的候选框都需要缩放到固定的大小。下面我们讲解要怎么进行缩放处理，为了简单起见我们假设下一阶段CNN所需要的输入图片大小是个正方形图片227*227。因为我们经过selective search 得到的是矩形框，paper试验了两种不同的处理方法：

**(1)各向异性缩放**

这种方法很简单，就是不管图片的长宽比例，管它是否扭曲，进行缩放就是了，全部缩放到CNN输入的大小227*227，如下图(D)所示；

**(2)各向同性缩放**

因为图片扭曲后，估计会对后续CNN的训练精度有影响，于是作者也测试了“各向同性缩放”方案。这个有两种办法

**A.** 直接在原始图片中，把bounding box的边界进行扩展延伸成正方形，然后再进行裁剪；如果已经延伸到了原始图片的外边界，那么就用bounding box中的颜色均值填充；如下图(B)所示;

**B.** 先把bounding box图片裁剪出来，然后用固定的背景颜色填充成正方形图片(背景颜色也是采用bounding box的像素颜色均值),如下图(C)所示;

![iou1](/images/blog/rcnn5.png)

对于上面的异性、同性缩放，文献还有个padding处理，上面的示意图中第1、3行就是结合了padding=0,第2、4行结果图采用padding=16的结果。经过最后的试验，作者发现采用各向异性缩放、padding=16的精度最高。

上面处理完后，可以得到指定大小的图片，因为我们后面还要继续用这2000个候选框图片，继续训练CNN、SVM。然而人工标注的数据一张图片中就只标注了正确的bounding box，我们搜索出来的2000个矩形框也不可能会出现一个与人工标注完全匹配的候选框。因此我们需要用IOU为2000个bounding box打标签，以便下一步CNN训练使用。在CNN阶段，如果用selective search挑选出来的候选框与物体的人工标注矩形框的重叠区域IoU大于0.5，那么我们就把这个候选框标注成物体类别，否则我们就把它当做背景类别。

### 2.2 网络设计

 网络架构我们有两个可选方案：第一选择经典的Alexnet；第二选择VGG16。经过测试Alexnet精度为58.5%，VGG16精度为66%。VGG这个模型的特点是选择比较小的卷积核、选择较小的跨步，这个网络的精度高，不过计算量是Alexnet的7倍。后面为了简单起见，我们就直接选用Alexnet，并进行讲解；Alexnet特征提取部分包含了5个卷积层、2个全连接层，在Alexnet中p5层神经元个数为9216、 f6、f7的神经元个数都是4096，通过这个网络训练完毕后，最后提取特征每个输入候选框图片都能得到一个4096维的特征向量。

#### 2.2.1 网络初始化

 直接用Alexnet的网络，然后连参数也是直接采用它的参数，作为初始的参数值，然后再fine-tuning训练。

 网络优化求解：采用随机梯度下降法，学习速率大小为0.001；

#### 2.2.2 fine-tuning阶段

  我们接着采用selective search 搜索出来的候选框，然后处理到指定大小图片，继续对上面预训练的cnn模型进行fine-tuning训练。假设要检测的物体类别有N类，那么我们就需要把上面预训练阶段的CNN模型的最后一层给替换掉，替换成N+1个输出的神经元(加1，表示还有一个背景)，然后这一层直接采用参数随机初始化的方法，其它网络层的参数不变；接着就可以开始继续SGD训练了。开始的时候，SGD学习率选择0.001，在每次训练的时候，我们batch size大小选择128，其中32个正样本、96个负样本。




## 三 Fast RCNN

### 3.1 引入原因

 FRCNN针对RCNN在训练时是multi-stage pipeline和训练的过程中很耗费时间空间的问题进行改进。它主要是将深度网络和后面的SVM分类两个阶段整合到一起，使用一个新的网络直接做分类和回归。主要做以下改进:

1. 最后一个卷积层后加了一个ROI pooling layer。ROI pooling layer首先可以将image中的ROI定位到feature map，然后是用一个单层的SPP layer将这个feature map patch池化为固定大小的feature之后再传入全连接层。

2. 损失函数使用了多任务损失函数(multi-task loss)，将边框回归直接加入到CNN网络中训练。

### 3.2 模型

fast rcnn 的结构如下

![fast rcnn结构](/images/blog/rcnn7.png)

图中省略了通过ss获得proposal的过程，第一张图中红框里的内容即为通过ss提取到的proposal，中间的一块是经过深度卷积之后得到的conv feature map，图中灰色的部分就是我们红框中的proposal对应于conv feature map中的位置，之后对这个特征经过ROI pooling layer处理，之后进行全连接。在这里得到的ROI feature vector最终被分享，一个进行全连接之后用来做softmax回归，用来进行分类，另一个经过全连接之后用来做bbox回归。

**注意：** 对中间的Conv feature map进行特征提取。每一个区域经过RoI pooling layer和FC layers得到一个 **固定长度** 的feature vector(这里需要注意的是，输入到后面RoI pooling layer的feature map是在Conv feature map上提取的，故整个特征提取过程，只计算了一次卷积。虽然在最开始也提取出了大量的RoI，但他们还是作为整体输入进卷积网络的，最开始提取出的RoI区域只是为了最后的Bounding box 回归时使用，用来输出原图中的位置)。


### 3.3 SPP网络

何恺明研究员于14年撰写的论文，主要是把经典的Spatial Pyramid Pooling结构引入CNN中，从而使CNN可以处理任意size和scale的图片；这中方法不仅提升了分类的准确率，而且还非常适合Detection，比经典的RNN快速准确。

本文不打算详细解释SPP网络，只介绍其中的SPP-layer，由于fast rcnn会使用到SPP-layer。

**SPP layer**

根据pooling规则，每个pooling   bin（window）对应一个输出，所以最终pooling后特征输出由bin的个数来决定。本文就是分级固定bin的个数，调整bin的尺寸来实现多级pooling固定输出。

如图所示，layer-5的unpooled FM维数为16*24，按照图中所示分为3级，

![fast rcnn结构](/images/blog/rcnn8.png)

第一级bin个数为1，最终对应的window大小为16*24；

第二级bin个数为4个，最终对应的window大小为4*8

第三级bin个数为16个，最终对应的window大小为1*1.5（小数需要舍入处理）

通过融合各级bin的输出，最终每一个unpooled FM经过SPP处理后，得到了1+4+16维的SPPed FM输出特征，经过融合后输入分类器。

这样就可以在任意输入size和scale下获得固定的输出；不同scale下网络可以提取不同尺度的特征，有利于分类。


### 3.4  RoI pooling layer

每一个RoI都有一个四元组（r,c,h,w）表示，其中（r，c）表示左上角，而（h，w）则代表高度和宽度。这一层使用最大池化（max pooling）来将RoI区域转化成固定大小的H*W的特征图。假设一个RoI的窗口大小为h*w,则转换成H*W之后，每一个网格都是一个h/H * w/W大小的子网，利用最大池化将这个子网中的值映射到H*W窗口即可。Pooling对每一个特征图通道都是独立的，这是SPP layer的特例，即只有一层的空间金字塔。

### 3.5 从预训练的网络中初始化数据

有三种预训练的网络：CaffeNet，VGG_CNN_M_1024，VGG-16，他们都有5个最大池化层和5到13个不等的卷积层。用他们来初始化Fast R-CNN时，需要修改三处：

①最后一个池化层被RoI pooling layer取代

②最后一个全连接层和softmax被替换成之前介绍过的两个兄弟并列层

③网络输入两组数据：一组图片和那些图片的一组RoIs

### 3.6 检测中的微调

使用BP算法训练网络是Fast R-CNN的重要能力，前面已经说过，SPP-net不能微调spp层之前的层，主要是因为当每一个训练样本来自于不同的图片时，经过SPP层的BP算法是很低效的（感受野太大）. Fast R-CNN提出SGD mini_batch分层取样的方法：首先随机取样N张图片，然后每张图片取样R/N个RoIs  e.g.  N=2 and R=128
除了分层取样，还有一个就是FRCN在一次微调中联合优化softmax分类器和bbox回归，看似一步，实际包含了多任务损失（multi-task loss）、小批量取样（mini-batch sampling）、RoI pooling层的反向传播（backpropagation through RoI pooling layers）、SGD超参数（SGD hyperparameters）。



## 4 Faster RCNN

Faster R-CNN统一的网络结构如下图所示，可以简单看作RPN网络+Fast R-CNN网络。

![fast rcnn结构](/images/blog/rcnn9.png)

原理步骤如下:

1. 首先向CNN网络【ZF或VGG-16】输入任意大小图片；

2. 经过CNN网络前向传播至最后共享的卷积层，一方面得到供RPN网络输入的特征图，另一方面继续前向传播至特有卷积层，产生更高维特征图；

3. 供RPN网络输入的特征图经过RPN网络得到区域建议和区域得分，并对区域得分采用非极大值抑制【阈值为0.7】，输出其Top-N【文中为300】得分的区域建议给RoI池化层；

4. 第2步得到的高维特征图和第3步输出的区域建议同时输入RoI池化层，提取对应区域建议的特征；

5. 第4步得到的区域建议特征通过全连接层后，输出该区域的分类得分以及回归后的bounding-box。

### 4.1 单个RPN网络结构

单个RPN网络结构如下:

![fast rcnn结构](/images/blog/rcnn10.png)

**注意：** 上图中卷积层/全连接层表示卷积层或者全连接层，作者在论文中表示这两层实际上是全连接层，但是网络在所有滑窗位置共享全连接层，可以很自然地用n×n卷积核【论文中设计为3×3】跟随两个并行的1×1卷积核实现






















## 参考博客

[RCNN 介绍](http://blog.csdn.net/hjimce/article/details/50187029)

[Fast RCNN介绍](http://blog.csdn.net/shenxiaolu1984/article/details/51036677)

[Faster RCNN论文笔记](http://blog.csdn.net/qq_17448289/article/details/52871461)

[Fast RCNN简要笔记](http://www.cnblogs.com/RayShea/p/5568841.html)

[SPP网络](http://www.itdadao.com/articles/c15a296465p0.html)
