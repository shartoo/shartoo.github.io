---
layout: post
title: 李宏毅深度学习-八-RNN和GAN
description: 李宏毅深度学习笔记
category: 深度学习
mathjax: true
---
## 1 生成网络

### 1.1 生成文本

我们需要知道的是，一个句子由字符或者单词组成。但是中文里面一个单词是一个有意义的单位。使用RNN生成句子时，每次由RNN生成一个字符或单词。

假若我们要基于RNN生成一个句子，生成过程如下：

+ 向RNN中输入 `<BOS>`(begin of sentence)
+ RNN会给出某个字符(或单词)的分布概率，根据这个概率分布抽样，可以得到一个词。比如说`床`
+ 继续把`床`这个词输入到RNN，RNN继续输出字符（或词）的分布概率，根据其概率分布抽样得到下一个词，以此类推，下一个词比如为`前`

![](/images/rnn_and_gan1.jpg)

### 1.2 生成图像

图像都是由像素组成，可以由RNN每次生成一个像素。图像的二维结构转换为一个像素序列，如下图：


![](/images/rnn_and_gan2.jpg)

序列处理过程如下：

![](/images/rnn_and_gan3.jpg)

但是这种方式只有像素值的连续性，没有考虑图像中像素的局部关联性。一种更贴近的方式是下面的图，中间黑色像素块同时受上面的红色块以及左边的粉红色块的影响。

![](/images/rnn_and_gan4.jpg)

但是要生成这种考虑空间特征的图像，可以使用Grid LSTM。如下图

![](/images/rnn_and_gan5.jpg)

左下角黑框为一个filter。最开始时输入一个类似`BOS`的字符，由Grid LSTM生成第一个像素蓝色块，再将蓝色块输入到Grid LSTM(中间图第二个黑色箭头)，Grid LSTM会生成第二个红色块，第二个红色像素块在生成时会考虑输入`蓝色块`和之前的输入信息。

如何产生空间信息，如下图

![](/images/rnn_and_gan6.jpg)

上图中，注意红色箭头的位置，是一个空间中的第二层。

**此方法可以生成 state of art 的图像。**


## 2  基于条件的生成

RNN只能生成一些随机的句子，我们想要基于条件生成想要的文本。比如在 image caption中，看到特定图像能生成对应的文本描述，在chatbot对话中会根据对话者所说的话来生成回应。

### 2.1 看图说话 image caption

一般使用 `CNN+RNN`的方式生成，使用cnn来抽取图像特征，再将特征传入RNN即可生成对应描述，如下图示例。

![](/images/rnn_and_gan7.jpg)

左边的CNN会将抽取出的特征vector传给右边的RNN，如果觉得只在开始的时候传入会导致RNN后续遗忘，可以每次都传入图像特征vector。


### 2.2 机器翻译

比如，我们想把中文`机器学习`翻译成对应的英文`machine learning`，即 `机器学习`$\Rightarrow$`machine learning`。此时二者之间毫无关联，但是我们把中文变成一个vector然后传入RNN。

**将中文变成vector**

同样可以使用RNN来完成，下图将`机器学习`这四个字用RNN抽取出一个vector代表整个句子的信息。

![](/images/rnn_and_gan8.jpg)

然后再将抽取出的vector传入给另外一个RNN，如下图：

![](/images/rnn_and_gan9.jpg)

其中红色的矩形块代表了中文部分抽取出的特征，可以三次重复传入右边的RNN，让右边的RNN分别输出对应的`machine`,`learning`以及句号(代表结束)。

比如可以同样将类似的方法来做chatbot，比如我输入一句`你好吗`，用RNN生成一个vector然后传入右边的RNN，让其生成`我很好`，类似的对话。

类似的设计在深度学习里面称之为**Encoder-Decoder**

![](/images/rnn_and_gan9.jpg)

它们二者可以联合训练。至于左边的encoder和右边的decoder是否一样，可以视情况而定，**encoder和decoder可以一样，也可以不一**。如果二者一样，可能容易导致过拟合。



## 3 Attention(Dynamic  Conditional Generation)

在上一节里面的`机器学习`$\Rightarrow$`machine learning`时，左边的encoder每次传入到右边的decoder都是**同样的vector**。其实，我们可以使得每次传给右边的vector不一样。比如，我们想要右边的输出为`machine`时，关注左边的`机器`即可，此时RNN应该可以更好的掌握

![](/images/rnn_and_gan11.jpg)

### 3.1  机器翻译

**基于注意力的模型**

+ **输入**： 每个时间点的每次词汇都可以用一个vector来表示，这个vector是RNN hidden layer的输出
+ **初始参数$z^0$**： 有一个初始的vector $z^0$，可以当做RNN的一个参数，它可以根据训练数据集学习得到
+ **匹配函数match**: 假设有这样一个匹配函数`match`，此函数由自己设计。用来计算每个timestep，输入词汇与$z^n$的匹配程度。match的示例
  + 余弦相似度: match函数可以是余弦相似度，来计算$z$和$h$ 的相似度。
  + 更小的神经网络：match函数可以是另外一个神经网络，输入是$z$和$h$,输出是一个标量（衡量匹配度）
  + 参数式: match函数里面有个矩阵参数$w$，此参数可以被学习。可以用$\alpha = h^TWz$来衡量匹配度。

例如：

![](/images/rnn_and_gan12.jpg)

现在对所有的RNN的隐藏层输出$h^t$计算与$z^0$的匹配度，然后加个softmax（非必要），得到所有输入词汇的匹配程度，如下：

![](/images/rnn_and_gan13.jpg)

经过匹配函数match之后，各个输入词汇的在$c^0$中所占比最发生变化，再将输出的$c^0$作为输入传给右边（decoder）的RNN，会使得decoder更加关注此示例中的`机器`这个单词，更容易学习输出对应的`machine`，如下图

![](/images/rnn_and_gan14.jpg)

可以将此步骤右边decoder的rnn的hidden layer的输出$z^1$作为左边encoder的新的匹配函数（当然可以是其他各式各样的方法），来继续下一步的输出。

![](/images/rnn_and_gan15.jpg)

### 3.2 语音识别

输入一段音频信号，可以将其抽取为一排vector，每个时间点大概0.01秒用一个vector来表示。神经网络会对所有的vector先计算匹配度，下图的下半部分黑色方格代表匹配度，颜色越深代表越匹配。

![](/images/rnn_and_gan16.jpg)

如图，使用第一个红色方格标记的部分黑色方格代表此时RNN需要注意的输入，将此输入传入给一个decoder，会得到对应的输出，即左边横轴的音素`h`，不仅如此，decoder还会产生空白。










































