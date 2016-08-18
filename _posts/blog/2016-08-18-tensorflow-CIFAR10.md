---
layout: post
title: tensorflow:使用tensorflow构建简单卷积神经网络
description: tensorflow
category: blog
---

本文翻译自: [使用tensorflow构建简单卷积神经网络](https://www.tensorflow.org/versions/r0.10/tutorials/deep_cnn/index.html)

# 一 概要

CIFAR-10分类问题是机器学习领域的一个通用基准，其问题是将32X32像素的RGB图像分类成10种类别:`飞机`，`手机`，`鸟`，`猫`，`鹿`，`狗`，`青蛙`，`马`，`船`和`卡车`。
更多信息请移步[CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)和[Alex Krizhevsky的演讲报告](http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)

# 二 目标

本教程的目标是建立一个相对简单的CNN卷积神经网络用以识别图像。在此过程中，本教程:
1. 高亮网络架构，训练和验证的典型组织。
2. 为构建更大更复杂的模型提供模板。

选择 CIFAR-10的原因是其复杂程度足以训练tensorFlow拓展成超大模型能力的大部分。同时，于训练而言，此模型也足够小，这对于想实现新想法和尝试新技术堪称完美。

# 三 教程的高亮部分

  CIFAR-10教程演示了使用TensorFlow构建更大更复杂模型几个重要结构：
  + 核心数学组件包括`卷积`,`修正的线性激活`,`最大池化`,`LRN即局部响应一体化`（AlexNet的论文3.3章）
  + 训练期间网络活动的可视化，包括输入图像、损失函数值、激活函数值和梯度的分布。
  + 计算学习参数的均线（moving average）的惯常做法，以及在评估期间使用这些均线来促进预测性能。
  + 随时间系统递减的学习速率清单的实现
  + 为消除从磁盘读取模型的延迟和代价极大的图像预处理的预取队列。

 我们同时提供了一个[多GPU版本](https://www.tensorflow.org/versions/r0.10/tutorials/deep_cnn/index.html#training-a-model-using-multiple-gpu-cards)的模型演示：
 + 配置并行环境下跨多个GPU的训练的单个模型
 + 在多GPU中共享和更新变量

 我们期望此教程能为基于TensorFlow的可视化任务创建更大CNNs模型提供起点。

 # 四 模型架构

  CIFAR-10教程中的模型是一个由可选的卷积和非线性层组成的多层架构。这些网络层之后是连接到一个softmax分类器的全连接的网络层。模型遵照Alxe Krizhevsky所描述的模型架设，只是最前面的几层有细微差别。
   此模型获得了一个极致的表现，在一个GPU上训练几个小时可以达到约86%的准确率。下文和代码部分有详细说明。模型由1068298个可学习的参数并且单张图片需要195000000个加乘操作以计算推理。

# 五 代码组织

 此教程的代码位于 [tensorflow/models/image/cifar10](https://www.tensorflow.org/code/tensorflow/models/image/cifar10/)

 文件|目标
 ---|---
 cifar10_input.py|读取本地的CIFAR-10二进制文件
 cifar10.py|创建CIFAR-10模型
 cifar10_train.py|在CPU或者GPU上训练CIFAR-10模型
 cifar10_multi_gpu_train.py|在多个GPU上训练CIFAR-10模型
 cifar10_eval.py	|评估CIFAR-10模型的预测性能

 # 六 CIFAR-10 模型

  CIFAR-10网络大部分代码在 `cifar10.py`。完整的训练图包括大概765个操作，我们发现通过使用以下模块来构建计算图，我们可以最大限度的重用代码:
 1. **模型输入**: inputs() 和 distorted_inputs() 分别是评估和训练的读取并预处理CIFAR图像的加法操作
 2. **模型预测**: inference()是推理加法操作，即在提供的图像中进行分类。
 3. **模型训练**：: loss() 和 train() 的加法操作是用来计算损失函数，梯度，变量更新和可视化摘要。

 # 七 模型输入

  模型的输入部分由从CIFAR-10的二进制文件读取函数 inputs()和distorted_inputs() 完成。这些文件包括固定字节长度的记录，因此我们使用`tf.FixedLengthRecordReader` 。查看**读取数据**来学习*Reader class*如何实现。
  图像将按照以下步骤进行处理：
  + 它们被裁减成24x24的像素，中央部分用来做评估或随机地用以训练。
  + 它们是[ approximately whitened](https://www.tensorflow.org/versions/r0.10/api_docs/python/image.html#per_image_whitening) 用以使模型对动态变化不敏感。

  对于训练部分，我们会额外应用一系列随机扭曲来人为增加数据集：
  + 将图像随机的左右翻转[随机翻转](https://www.tensorflow.org/versions/r0.10/api_docs/python/image.html#random_flip_left_right)
  + 随机扰乱图像亮度[随机亮度](https://www.tensorflow.org/versions/r0.10/api_docs/python/image.html#random_brightness)
  + 随机扰乱图像对比度[随机对比度](https://www.tensorflow.org/versions/r0.10/api_docs/python/image.html#random_contrast)

在[图像页](https://www.tensorflow.org/versions/r0.10/api_docs/python/image.html)可以查看可用的扭曲方法列表。为了在TensorBoard中可视化，我们也在图像中增加了一个 [缩略图](https://www.tensorflow.org/versions/r0.10/api_docs/python/image.html)。这对于校验输入是否构建正确是个良好举措。
![](/images/blog/cifar_image_summary.png)
从磁盘中读取图像并扰乱，可能会消耗不确定的时间。为防止这些操作拖慢训练过程，我们使用16个独立线程不断地从一个TensorFlow队列。

# 八 模型预测

 模型的预测部分由*inference()*函数构建，该操作会添加其他操作以计算预测逻辑。模型的此部分组织如下：

 网络层名称| 描述
 -----|------
 conv1|卷积和修正线性激活层
 pool1|最大池化
 norm1|局部响应一体化
 conv2|卷积和修正线性激活层
 norm2|局部响应一体化
 pool2|最大池化
 local3|使用修正线性激活的全连接层
 local4|使用修正线性激活的全连接层
 softmax_linear|线性转换以产生logits

 下图由**TensorBoard**生成用以描述推理操作
 ![推理操作](/images/blog/cifar_graph.png)

 *input()*和*inference()* 函数提供了在模型上进行评估的全部所需组件。我们先将注意力移到训练模型。

 # 九 模型训练
  训练一个完成N类分类的网络的常用方法是多项式logstic回归,aka.softmax回归。softmax回归应用一个 [softmax](https://www.tensorflow.org/versions/r0.10/api_docs/python/nn.html#softmax)
