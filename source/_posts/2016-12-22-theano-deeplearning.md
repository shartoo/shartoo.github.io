---
layout: post
title: theano 深度学习数据准备
description: 深度学习
category: 深度学习
mathjax: true
---

## 数据集

 使用的是MNIST手写字识别,[下载](http://deeplearning.net/data/mnist/mnist.pkl.gz)。数据由60000个样本，50000个训练样本和10000个测试样本，每个样本被统一为28x28像素。原始数据集中每个样本的像素值区间是[0,255]，其中0代表黑色，255为白色，中间的是灰色。

 ![样本](/images/blog/theano1.png)![样本](/images/blog/theano2.png)![样本](/images/blog/theano3.png)![样本](/images/blog/theano4.png)


## 卷积神经网络

 CNN每层都是由多个特征map组成，( $h^{()k},k=0...k$ )。隐藏层的权重可以用一个四维的Tensor表示，分别代表**目标特征map**，**源特征map**，**原始图像中竖轴坐标**，**原始图像中横轴坐标**。偏置b可以用一个向量表示，向量中每个元素是目标特征map中的一个元素。下图演示了这个过程:

 ![CNN](/images/blog/theano5.png)

上图展示了两层CNN网络，`m-1`层包含了4个特征map，隐藏层`m`包含了两个特征map ( $h^0和h^1$ )。 输出神经元中的像素，$h^0$ 和 $h^1$ (途中蓝色( $W^1$ 旁边)和红色矩形框( $W^2$ 旁边))，是从 `m-1` 层的2x2的接收域的像素中计算得来的。注意看，接受域是如何覆盖全部的四个输入特征map的。特征 $h^0$ 和 $h^1$ 的权重 $W^0$ 和 $W^1$ 都是3维Tensor,第一个维度代表的是第几个输入特征maps的，其他两个是像素坐标。

结合起来看 $W^{kl} _{ij}$ 代表第 `m`层的第 `k` 个特征的每个像素与第 `m-1`层的第 `l` 个特征的坐标点为 `i,j` 像素的权重。所以，在上图由4个通道卷积得到2个通道的过程中，参数的数目为4×2×2×2个，其中4表示4个通道，第一个2表示生成2个通道，最后的2×2表示卷积核大小。

下面演示 theano的代码和详细数据过程：

**导入基本包**

```
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d

import numpy

```

**初始化一个随机数生成器**

```
 rng = numpy.random.RandomState(23455)

```

**初始化输入**

注意，深度学习框架中，一般会先构架计算图（即，把计算过程定义好），然后再输入数据。所以，此时的初始化时一个空的占位变量。

```
# instantiate 4D tensor for input
input = T.tensor4(name='input')
```

通过调试，查看变量的初始化
![CNN](/images/blog/theano6.png)

**初始化权重矩阵**

注意：权重的数据类型是 `theano.shared`

```
# initialize shared variable for weights.
w_shp = (2, 3, 9, 9)
w_bound = numpy.sqrt(3 * 9 * 9)
W = theano.shared( numpy.asarray(
            rng.uniform(
                low=-1.0 / w_bound,
                high=1.0 / w_bound,
                size=w_shp),
            dtype=input.dtype), name ='W')
```
观察权重的数据结构和内部的值。

![CNN](/images/blog/theano7.png)

**初始化偏置**

偏置只有2个，因为此处只使用了 2个特征抽取map。

此时的偏置的数据类型也是 `theano.shared`

```
b_shp = (2,)
b = theano.shared(numpy.asarray(
           rng.uniform(low=-.5, high=.5, size=b_shp),
           dtype=input.dtype), name ='b')
```

**计算卷积**

构建符号表达的计算图，此时是个空架子

```
# build symbolic expression that computes the convolution of input with filters in w
conv_out = conv2d(input, W)
```

![CNN](/images/blog/theano8.png)

**卷积网络的激活输出**


```
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

# create theano function to compute filtered images
f = theano.function([input], output)
```

** 给卷积网络灌入图像数据**

下面的代码直接使用了CNN网络中的函数 `f`，注意下面代码中的 `filtered_img = f(img_)`，其他部分是普通的图像处理。此处往图像中灌入图像数据，并返回卷积之后的结果。

```
import numpy
import pylab
from PIL import Image

# open random image of dimensions 639x516
img = Image.open(open('doc/images/3wolfmoon.jpg'))
# dimensions are (height, width, channel)
img = numpy.asarray(img, dtype='float64') / 256.

# put image in 4D tensor of shape (1, 3, height, width)
img_ = img.transpose(2, 0, 1).reshape(1, 3, 639, 516)
filtered_img = f(img_)

# plot original image and first and second components of output
pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray();
# recall that the convOp output (filtered image) is actually a "minibatch",
# of size 1 here, so we take index 0 in the first dimension:
pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])
pylab.show()
```

下图演示了图片被输入到图像中的结构:

图片被处理为`ndarray`类型，并保留了RGB通道的结构。

![CNN](/images/blog/theano9.png)

**详细的卷积过程**

查看函数 `conv_out=conv2d(input,w)` 在`theano`中如何定义：

```
def conv2d(input, filters, input_shape=None, filter_shape=None,
           border_mode='valid', subsample=(1, 1), filter_flip=True,
           image_shape=None, filter_dilation=(1, 1), **kwargs):

    if 'imshp_logical' in kwargs or 'kshp_logical' in kwargs:
        raise ValueError(
            "Keyword arguments 'imshp_logical' and 'kshp_logical' for conv2d "
            "are not supported anymore (and have not been a reliable way to "
            "perform upsampling). That feature is still available by calling "
            "theano.tensor.nnet.conv.conv2d() for the time being.")
    if len(kwargs.keys()) > 0:
        warnings.warn(str(kwargs.keys()) +
                      " are now deprecated in "
                      "`tensor.nnet.abstract_conv.conv2d` interface"
                      " and will be ignored.",
                      stacklevel=2)

    if image_shape is not None:
        warnings.warn("The `image_shape` keyword argument to "
                      "`tensor.nnet.conv2d` is deprecated, it has been "
                      "renamed to `input_shape`.",
                      stacklevel=2)
        if input_shape is None:
            input_shape = image_shape
        else:
            raise ValueError("input_shape and image_shape should not"
                             " be provided at the same time.")

    return abstract_conv2d(input, filters, input_shape, filter_shape,
                           border_mode, subsample, filter_flip,
                           filter_dilation)
```

看代码可知，此函数只是个外壳，最后的`abstract_conv2d`才是实际的处理方法。继续进入到 `abstract_conv2d`函数内部：

```
def conv2d(input,
          filters,
          input_shape=None,
          filter_shape=None,
          border_mode='valid',
          subsample=(1, 1),
          filter_flip=True,
          filter_dilation=(1, 1)):
   """This function will build the symbolic graph for convolving a mini-batch of a
   stack of 2D inputs with a set of 2D filters. The implementation is modelled
   after Convolutional Neural Networks (CNN).

   Refer to :func:`nnet.conv2d <theano.tensor.nnet.conv2d>` for a more detailed documentation.
   """

   input = as_tensor_variable(input)
   filters = as_tensor_variable(filters)
   conv_op = AbstractConv2d(imshp=input_shape,
                            kshp=filter_shape,
                            border_mode=border_mode,
                            subsample=subsample,
                            filter_flip=filter_flip,
                            filter_dilation=filter_dilation)
   return conv_op(input, filters)
```

此函数，也只将python中的`input`变量和`filters`变量转换为`theano` 中的张量Tensor。之后将初始化了一个`AbstractConv2d`对象

```

   input = as_tensor_variable(input)
   filters = as_tensor_variable(filters)
   conv_op = AbstractConv2d(imshp=input_shape,
                             kshp=filter_shape,
                             border_mode=border_mode,
                             subsample=subsample,
                             filter_flip=filter_flip,
                             filter_dilation=filter_dilation)
    return conv_op(input, filters)
```

继续深入 `AbstractConv2d`这个构造方法内部

```

class AbstractConv2d(AbstractConv):
    """ Abstract Op for the forward convolution.
    Refer to :func:`BaseAbstractConv <theano.tensor.nnet.abstract_conv.BaseAbstractConv>`
    for a more detailed documentation.
    """

    def __init__(self,
                 imshp=None,
                 kshp=None,
                 border_mode="valid",
                 subsample=(1, 1),
                 filter_flip=True,
                 filter_dilation=(1, 1)):
        super(AbstractConv2d, self).__init__(convdim=2,
                                             imshp=imshp, kshp=kshp,
                                             border_mode=border_mode,
                                             subsample=subsample,
                                             filter_flip=filter_flip,
                                             filter_dilation=filter_dilation)
```

从代码中可以看到，它实际是`super(AbstractConv2d, self)`继续调用父类`AbstractConv`的`super(AbstractConv2d, self)`的构造方法。但是`theano`中此处的父类是个抽象类，它在执行```super(AbstractConv2d, self)```时，实际是调用了自己的构造方法。
