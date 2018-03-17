---
layout: post
title: 李宏毅深度学习-六-HighwayNetwork和LSTM
description: 李宏毅深度学习笔记
category: blog
---

## 1   前馈网络和循环神经网络(RNN)

**前馈网络示意图**

![前馈网络示意图](/images/blog/LHY_RNN1.jpg)    


上图中，$f_1,f_2,f_3,f_4..$表示的是前馈网络的网络层，一个输入$x$，一个输出$y$。

**循环神经网络**

![RNN示意图](/images/blog/LHY_RNN2.jpg)    

对比可知，两者十分相似。不同之处是，

1. 循环神经网络每个网络层都有输入，而前馈网络只有一个输入
2. 循环神经网络的每一层的激活函数是同一个，而前馈网络的每一层的激活函数都不同。

## 2 如何把GRU改成Highway Network

GRU网络示意图如下：

![GRU示意图](/images/blog/LHY_GRU.jpg)   

变成  

![GRU示意图](/images/blog/LHY_GRU2.jpg)    

1. 拿掉每个时间步的输入$x^t$,
2. 拿掉$y^t$,RNN每个时间步都会输出一个$y^t$,Highway network只有一个最后的输出$y^t$.
3. 把 $h^{t-1}$ 改成 $a^{t-1}$。其中$a^{t-1}$ 是第$t$层的输出
4. 拿掉`reset gate`。它的作用是让GRU忘记之前发生过的事情，但是Highway不应该忘记，它只有一个开始的。

## 3 Highway Network

一个常见的Highway Network的示意图如下：

![Highway示意图](/images/blog/LHY_Highway.jpg)   

注意，它由两部分组成：Gate controller 部分和copy部分。其中的$z,h^{'},a^{t-1}$（可以通过第二节看到这些详细的）分别如下计算：
$$
 h^{'}=\sigma (Wa^{t-1}) \\
z=\sigma(W^{'}a^{t-1}) \quad\quad  蓝色部分\\
a^t = z\bigodot a^{t-1}+(1-z)\bigodot h  \quad\quad  黑色部分
$$

一个较深的Highway Network示意图

![Highway示意图](/images/blog/LHY_Highway2.jpg)   

它在训练过程中会自动给连接层之间的gate 赋予权重，会自动丢弃某些不重要的层，会自动决定需要多少层。

事实上Highway Network的论文中有论证，通过不断丢弃某些层来评估对网络loss的影响。下图是在`MNIST`数据集上评测`ResNet`网络，下图中横轴代表的是网络层，纵轴代表的是网络loss。可以看到$15~45$层 这些层被丢弃之后对网络loss几乎没有影响。


![Highway示意图](/images/blog/LHY_Highway_loss1.jpg) 

另外一张图是评测`ResNet`在CIFAR-10数据集上的结果，可以看到拿掉某些层对网络性能的影响非常大。CIFAR-10是个比较复杂的数据集。

![Highway示意图](/images/blog/LHY_Highway_loss2.jpg) 

## 4 Grid LSTM

它是一种既横着，又竖着的LSTM。它既有时间方向的记忆，又有深度方向的记忆（左边是**LSTM**，右边是**Grid LSTM**）：

![GridLSTM示意图](/images/blog/LHY_GRIDLSTM1.jpg) 


原来的LSTM的输入是 $c$ 和 $h$，输出是$c'$和$h^t$，这些都是时间方向上的

Grid LSTM时间方向上与传统的LSTM一致，**多出了一个深度方向的输入输出.输入是$a$,$b$输出是$a'$,$b'$**

### 4.1 Grid LSTM如何连接

![GridLSTM示意图](/images/blog/LHY_GRIDLSTM2.jpg) 

### 4.2  Grid LSTM内部结构

![GridLSTM示意图](/images/blog/LHY_GRIDLSTM3.jpg) 

其中

+ $h$ 是**输入**
+ $z^f$ 是 **遗忘门**
+ $z^i$ 是**输入门**
+ $z$  是 **输入信息**
+ $z^o$ 是**输出门**
+ $c$  是**记忆**

我们 可以将上图右边做一个切分，分别是**历史记忆**，**当前输入**，**准备输出**

![GridLSTM示意图](/images/blog/LHY_GRIDLSTM4.jpg) 

### 4.3 Grid LSTM的输入输出

Grid LSTM 有**两套记忆**以及**两套隐藏层输出** ，如何结合并表现出来？

![GridLSTM示意图](/images/blog/LHY_GRIDLSTM5.jpg) 

其中 $h$和$b$一起产生各类门(`遗忘门`,`输出门`,`输入门`,`输入信息`)， $c$和$a$组合成一串较长向量作为历史记忆。


### 4.4 3D Grid LSTM


![GridLSTM示意图](/images/blog/LHY_3DGRIDLSTM.jpg) 










