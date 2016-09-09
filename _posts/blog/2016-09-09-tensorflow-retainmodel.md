---
layout:     post
title:      tensorflow：如何重新训练Inception模型的最后一层，以应对新分类
category: blog
description: tensorflow
---

本文整理自 (retrain network)[https://www.tensorflow.org/versions/r0.10/how_tos/image_retraining/index.html]

## 在Flowers数据集上重新训练

!(flowers数据集)[/images/blog/retain_flowers.jpg]

训练开始之前你需要一些数据集，以告诉网络，你有哪些新分类需要学习。后面的部分会道明如何准备自己的数据，首先按照如下操作下载一些数据集

```
cd ~
curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
tar xzf flower_photos.tgz
```
等图片下载好，可以按照如下方式重新训练，在tensorflow源码的根目录下执行:

```
bazel build tensorflow/examples/image_retraining:retrain
```
然后继续执行如下操作

```
bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir ~/flower_photos
```
注意：`image_dir`参数需要指定你的flowers数据下载目录。
此脚本会载入预训练的Inception v3模型，移除原先模型中的最后一层，然后在flowers数据集上重新训练。flowers数据集的分类信息在原始的ImageNet网络中并不存在 ，转换学习的神奇之处在于网络的前几层用来训练识别物体之间的差别，这可以在做任何更新的情况下重用于新的分类任务。

## Bottlenecks

  脚本运行可能消耗半小时或者更多，这取决于你的硬件。脚本的第一阶段会分析所有图像并计算每张图像的Bottleneck.`bottleneck`是我们经常用于描述网络最后一层之前的那些实际完成分类任务的网络层的一种非正式称谓。倒数第二层的输出结果对于描述区分需要分类的类别已经足够.这意味着它必须有信息丰富并且关于图像信息也足够紧凑，因为它必须包含足够的信息来在一小撮数值（标签值）中完成分类。重新训练最后一层就可以完成新的分类任务，为何？因为在ImageNet数据上完成1000个分类任务的信息通常也可用于区分新的物体。

  由于每张图像在训练和计算bottleneck值的过程中重复使用多次，这极为耗时，将这些数据缓存在磁盘上有助于加速整个过程以避免重复计算。

## 训练

 一旦bottleneck完成，网络最后一层的训练就开始了。你将会看到训练步骤的一系列的输出，每一个会输出训练准确率，验证准确率和交叉熵。训练准确率显示的当前批次的图像中正确分类的准确率。验证准确率显示的是从一组不同数据中随机选择的一组图像的精度。这其中的关键区别在于，训练的准确率是基于网络已经能够学习的图像集，因为网络
