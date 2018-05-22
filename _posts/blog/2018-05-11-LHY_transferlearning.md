---
layout: post
title: 李宏毅深度学习：迁移学习
description: 深度学习基础理论
category: blog
---


笔记视频：https://www.bilibili.com/video/av15889450/#page=26

## 1 基本概念

迁移学习：当前训练的数据集与目标任务没有直接相关。此处的不直接相关特指以下情形：

+ 相同领域，但是不同任务。比如都是对动物分类的，但是分类的目标不同，比如只有猫和狗的图片数据，但是分类任务是对大象和老虎分类。

![](/images/blog/transfer_learning1.jpg)
![](/images/blog/transfer_learning2.jpg)

+ 领域不同，但是任务相同。 比如同样是对猫和狗做分类，但是数据是真是相机拍摄的图像，而目标是招财猫和动漫狗，它们的数据分布不一致

![](/images/blog/transfer_learning3.jpg)


### 1.1 为何会考虑迁移学习

数据不充足的情况下，可能会考虑使用迁移学习，比如以下情况。

|目标领域|目标任务|不相关的数据|
|---|---|---|
|语音识别|对台湾语做识别|从youtube上爬取英文、中文语音数据训练模型来迁移学习|
|图像分类|医疗数据极度缺乏，做医疗诊断时|使用已有的海量图像数据(coco,imagenet等)|
|文本分析|特定领域，比如法律文件分析|可以从其他领域的文本分析迁移|

## 2 可以用迁移学习做什么

### 2.1 模型fine-tuning

当我们的特定任务所拥有的数据集非常少(比如识别某个人的声音，但是那个人的声音数据很少)，但是非相关的数据集很多(比如来自很多人的很多语音数据)，我们无法用某一个人的声音数据来训练一个语音识别模型，这种情况要做迁移学习，可以称之为**one-shot**。用许多人的语音数据训练模型，再来某个人的语音数据来fine-tuning。


**问题**

目标数据集太少，即便是用非直接相关数据训练出了一个初始模型，然后用目标数据集做迁移学习，很容易会导致**过拟合**。

## 3 迁移学习技巧

### 3.1 Conservation Training

例如，已经有大量的source data数据（比如语音识别中大量的不同speaker的语音数据），以及target data(某个speaker的语音数据)。此时如果直接用source data训练出来的模型，再用target data做迁移学习，模型可能就会坏掉。 

可以在training的时候，加一些限制(就是加一些非L1,L2的正则化)，使得训练完成之后，前后两次模型效果差不太多。

![](/images/blog/transfer_learning4.jpg)

### 3.2 Layer transfer

先用源数据训练出一个模型，然后将这个模型的某些层网络直接复制到新的网络中，然后只用新数据训练网络的余下层网络。这样训练时只需要训练很少的参数。

![](/images/blog/transfer_learning5.jpg)

但是，哪些层应该被transfer，哪些不应该被transfer? 不同的任务之中，需要transfer的网络层不同。

+ 语音识别中，通常只复制最后几层网络。然后重新训练输入层网络。(同样的发音方式，得到的结果不同)语音识别的结果，应该跟发音者没有关系的，所以最后几层是可以被复制的。而不同的地方在于，从声音信号到发音方式，每个人都不一样。

+ 在图像任务中。通常只复制前面几层，而训练最后几层。通常前几层做的就是检测图像中有没有简单的几层图形，而这些是可以迁移到其他任务中。而通常最后几层通常是比较特异化的，这些是需要训练的。
   
![](/images/blog/transfer_learning6.jpg)

**网络层迁移学习的实验结果(图像任务)**

ImageNet的数据120万图像分为source和target，按照分类数量划分，其中500个分类作为source，另外500个为target。其中横轴为transfer learning复制的网络层，其中0代表没有复制网络层，纵轴为分类准确率。可以发现，当我们只复制前面几个网络层时，效果有提升，但是复制得太多效果就开始变差。

![](/images/blog/transfer_learning7.jpg)

上图中

+ 5 黄色线：代表在做了复制网络前几层之后，做了fine tuning之后的结果

+ 3蓝色线：对照组。在目标领域上训练出一个模型，然后复制此模型的前面几层，然后固定住这几层，接着继续用**目标数据**训练剩下的几层。

+ 2蓝色线：对照组。在目标领域上训练出一个模型，然后复制此模型的前面几层，然后固定住这几层，接着继续用**新数据*训练剩下的几层。结果，有时候很差。在训练时，前面的层和后面的层其实是需要相互搭配的，否则后面的层的结果就很差。

+ 4红色线：

**source和target不是同种分类数据时**

如果source和target是不同的分类数据，比如source数据是自然风光，而target是人造物体，那么做transfer learning时，其准确率会大幅度降低。

![](/images/blog/transfer_learning8.jpg)

如果只复制前面几层时，与没有复制没有太多区别。


### 3.3 多任务学习

一个成功的实例是，多语言语音识别。输入是不同语言的语音，前面的几层公用参数，后面的几层不同参数。

![](/images/blog/transfer_learning9.jpg)

## 4 progressive neural network

![](/images/blog/transfer_learning10.jpg)

先训练一个Task1的网络，训练完成之后，固定其参数。再去训练一个Task2，它的每个隐藏层的都会去接 Task1的隐藏层输出。它的好处是，即便Task1和Task2完全不像，Task2的数据不会影响到Task1的模型参数，所以迁移的结果一定不会更差，最糟糕的情况就是直接训练Task2模型(此时Task1的输出设置为0)





## 5 labeled source data  & unlabeled target data

源数据为标记数据$(x^s,y^s)$ 作为训练集，而目标数据为非标记数据$x^t$为测试集。比如下图的`MNIST`数据集为训练集，而`MNIST-M`为测试集，其中`MNIST-M`同样为手写字，不过其背景变为风景和彩色的。

![](/images/blog/transfer_learning11.jpg)

我们分析下领域对抗训练，把CNN作为特征抽取工具，会发现source data有很明显的分类现象，而target data却没有。

![](/images/blog/transfer_learning12.jpg)

如上图中，MNIST数据集很明显的分为10个团，而MNIST-M没有。此时对于MNIST-M无能为力。

所以，我们希望CNN的feature extractor能够消除领域特性，就需要使用 `domain-adversarial training`

### 5.1 domain-adversarial training


![](/images/blog/transfer_learning13.jpg)

feature extractor与domain classifer做相反的事，domain classifer 极力区分当前数据的来源，而feature extractor希望domain classifer能够无视domain 的差异。

![](/images/blog/transfer_learning14.jpg)

其实际做法是，在计算BP时，feature extractor 将domain classifer的梯度乘以负1，然后传给 domain classifer。

以下是这种训练出来的实验结果

![](/images/blog/transfer_learning15.jpg)

## 6  zero-shot

即：测试集里面的分类数据是训练集中从未出现过的，比如训练集的分类是毛和狗，而测试集里面却有草泥马

![](/images/blog/transfer_learning16.jpg)

这种任务，在语音识别中很常见，训练集中不可能出现所有的语音和词汇。在语音上的做法是，不去直接辨别一段声音属于哪个word，而是去辨别一段声音属于哪个音标，然后做一个音标和tab之间的对应关系表，即lexics。所以，即便某些词汇没有出现在训练集，也可以从音标和lexics表得到。

那么，这个操作应用到图像中就是以每种分类的特定属性替代分类。比如狗这个分类，以`furry`,`4 legs`,`tail`这些属性来表示，切记这些属性必须足够丰富才可以。那么在训练时，就不直接识别分类，而是识别图像具备哪些属性。

![](/images/blog/transfer_learning17.jpg)

测试集的时候，即便来了一个未出现的动物，也可以使用这些属性描述。

![](/images/blog/transfer_learning18.jpg)

如果属性集太大，还可以做 attribute embedding。

![](/images/blog/transfer_learning19.jpg)

其中$f(x^2)和g(y^2)$ 都可能是神经网络。


### 6.1 zero-shot的成功应用

机器翻译

![](/images/blog/transfer_learning20.jpg)

在没有日语翻译成韩文的数据集，由于有韩语翻译到英文、日文和英文翻译到日文，所以可以完成从日语翻译成韩文。根据学习好的encoder，把各种语言的词汇映射到空间中的向量，会出现下图的结果

![](/images/blog/transfer_learning21.jpg)

上图中，不同颜色代表不同语言，处于相同位置的代表意义相同。
