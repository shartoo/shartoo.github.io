---
layout: post
title: 深度学习：batch_size的设置与影响
description: 神经网络基础
category: blog
---

注意：本文根据知乎**程引**的[回答](https://www.zhihu.com/question/32673260)整理

## 为何需要batch_size参数

 Batch的选择，**首先决定的是下降的方向**。如果数据集比较小，完全可以采用 **全数据集(Full Batch Learning)** 的形式。这样做有如下好处：

 + 全数据集确定的方向能够更好的代表样本总体，从而更准确的朝着极值所在方向。
 + 由于不同权值的梯度值差别较大，因此选取一个全局的学习率很困难。

 Full Batch Learning可以使用Rprop只基于梯度符号并且针对性单独更新各权值。
 但是对于非常大的数据集，上述两个好处变成了两个坏处：

 + 随着数据集的海量增加和内存限制，一次载入所有数据不现实。
 + 以Rprop的方式迭代，会由于各个batch之间的采样差异性，各此梯度修正值相互抵消，无法修正。这才有了后来的**RMSprop**的妥协方案。

 ## Full Batch Learning的另一个极端 Online Learning
 既然 Full Batch Learning 并不适用大数据集，那么走向另一个极端怎么样？所谓另一个极端，就是每次只训练一个样本，即 Batch_Size = 1。这就是**在线学习(Online Learning)** 。线性神经元在均方误差代价函数的错误面是一个抛物面，横截面是椭圆。对于多层神经元、非线性网络，在局部依然近似是抛物面。使用在线学习，每次修正方向以各自样本的梯度方向修正，横冲直撞各自为政，**难以达到收敛**

 ![batch_size](/images/blogs/batch_size1.png)

 ## 选取适中的batch_size

  可不可以选择一个适中的 Batch_Size 值呢？当然可以，这就是**批梯度下降法（Mini-batches Learning）**。因为如果数据集足够充分，那么用一半（*甚至少得多*）的数据训练算出来的梯度与用全部数据训练出来的梯度是**几乎一样**的。
  在合理范围内，增大 Batch_Size 有何好处？

  + 内存利用率提高了，大矩阵乘法的并行化效率提高。
  + 跑完一次 epoch（全数据集）所需的迭代次数减少，对于相同数据量的处理速度进一步加快。
  + 在一定范围内，一般来说 Batch_Size 越大，其确定的下降方向越准，引起训练震荡越小。

  盲目增大 Batch_Size 有何<u>坏处

  + 内存利用率提高了，但是内存容量可能撑不住了。
  + 跑完一次 epoch（全数据集）所需的迭代次数减少，要想达到相同的精度，其所花费的时间大大增加了，从而对参数的修正也就显得更加缓慢。
  + Batch_Size 增大到一定程度，其确定的下降方向已经基本不再变化。

##  调节 Batch_Size 对训练效果影响到底如何？

  这里跑一个 LeNet 在 MNIST 数据集上的效果。MNIST 是一个手写体标准库

   ![batch_size](/images/blogs/batch_size1.png)

  运行结果如上图所示，其中绝对时间做了标幺化处理。运行结果与上文分析相印证：
  + Batch_Size 太小，算法在 200 epoches 内不收敛。
  + 随着 Batch_Size 增大，处理相同数据量的速度越快。
  + 随着 Batch_Size 增大，达到相同精度所需要的 epoch 数量越来越多。
  + 由于上述两种因素的矛盾， Batch_Size 增大到<u>某个</u>时候，达到<b>时间上</b>的最优。
  + 由于最终收敛精度会陷入不同的局部极值，因此 Batch_Size 增大到<u>某些</u>时候，达到最终收敛**精度上**的最优。

## caffe中batch size影响

 caffe的代码实现上选取一个batch的时候似乎是按着数据库的图片顺序选取输入图片的，所以在生成数据库的时候切记要shuffle一下图片顺序。caffe中完成这一步的代码为

 ```
 $caffe_root/build/tools/convert_imageset -shuffle -resize_height=256 -resize_width=256
 ```
