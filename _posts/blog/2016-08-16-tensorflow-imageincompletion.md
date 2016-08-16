---
layout: post
title: tensorflow:使用深度学习完成受损图像修补
description: tensorflow
category: blog
---

# 一 文章结构

+ 入门简介
+ 第一步
  + 以概率分布解读图像抽样
  + 如何填补丢失的信息
  + 但是统计学应该拟合哪里？这些是图像
  + 那么，我们如何完成图像修补
+ 第二步：学习如何从概率分布中生成新样本
  + [ML-Heavy]生成对抗性网络(GAN)构建模块
  + 使用 $G(z)$ 产生假图像
  + [ML-Heavy] 训练 DCGANS
  + 存在 GAN和DCGANS实现
  + 在你自己的图片上运行 DCGAN
+ 第三步：找到修补图像的最合适的假图像
  + 使用DCGANs进行图像修补
  + [ML-Heavy]投影到 $p_g$上的损失函数
  + [ML-Heavy]使用DCGANs算法和tensorflow完成图像修补
  + 修补你的图像
+ 结论
+ 进一步阅读的参考书目
+ 在tensorflow和 Torch上未完成的想法
