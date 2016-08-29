---
layout: post
title: 神经网络的一些基本概念
description: 神经网络
category: blog
---

## 误差函数

  令$y_j(n)$记为在输出层第$j$个神经元输出产生的函数信号。相应的，神经元$j$输出所产生的误差信号定义为：
  $$
           e_j(n)=d_j(n)-y_j(n)
  $$
  其中$d_j(n)$是响应向量$d(n)$的第$j$个元素。那么**瞬时误差能量**(*instaneous error energy*)定义为:
  $$
      \Im _j(n)=\frac{1}{2}e^2_j(n)
  $$
  将所有输出层误差能量相加，得到整个网络的全部瞬时误差能量:
  $$
      \Im (n)=\sum _{j\epsilon C}\Im(n)=\frac{1}{2}\sum_{i\epsilon C}e^2_j(n)
  $$
其中集合C包括输出层的所有神经元。设训练样本中包含N个样例，训练样本上的**平均误差能量(error energy averaged over the training sample)**或者说经验风险(empirical risk)定义为:
$$
    \Im_{av}(N) = \frac{1}{N}\sum^N_{n=1}\Im (n)=\frac{1}{2N}\sum^N_{n=1}\sum_{j\epsilon C}e^2_j(n)
$$

## 批量学习

   在批量学习方法中，多层感知器的突出权值的调整在训练样本集合$\Im$的所有N个样本都出现后进行，这构成了训练的一个回合(epoch)。即**批量学习的代价函数是由平均误差能量\Im_{av}**定义的。多层感知器的突触权值的调整是以回合-回合为基础的(epoch-by-epoch)
