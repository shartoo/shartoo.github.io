---
layout: post
title: darknet在nvidia tx2上的训练自己的数据
description: 深度学习实战
category: blog
---

## 一 准备数据

**注意**：所有的文件最好在linux下通过代码或者vi的方式来创建，如果从window下创建再拷贝过去的话，很容易出现各种找不到文件的错误。

首先知道yolo需要的几个数据

+ cfg
+ data
+ names
+ weights

其中前三个内容分别可以为:

### 1.1 obj.cfg
yolo的网络配置文件，一般可以从其官网 [darknet-yolo-cfg文件](https://github.com/pjreddie/darknet/blob/master/cfg/yolo-voc.cfg)下载一份，然后修改。不要自己手动创建，容易因为编码问题导致程序无法运行。

内容为：

```
[net]
# Testing
batch=1
subdivisions=1
# Training
# batch=64
# subdivisions=8
height=416
width=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.001
burn_in=1000
max_batches = 80200
policy=steps
steps=40000,60000
scales=.1,.1

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky


#######

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[route]
layers=-9

[convolutional]
batch_normalize=1
size=1
stride=1
pad=1
filters=64
activation=leaky

[reorg]
stride=2

[route]
layers=-1,-4

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=125
activation=linear


[region]
anchors =  1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071
bias_match=1
classes=20
coords=4
num=5
softmax=1
jitter=.3
rescore=1

object_scale=5
noobject_scale=1
class_scale=1
coord_scale=1

absolute=1
thresh = .6
random=1
```

### 1.2 obj.data

个人的数据配置，内容如下:

```
classes= 1  
train  = train.txt  
valid  = test.txt  
names = obj.names  
backup = backup/
```
其中 

+ classes:为训练数据的类别数目，比如4分类模型，则为4
+ train: 训练集图片路径。一般是相对于darknet根目录的路径。
+ valid: 测试集图片路径。与train相同
+names: 图片类别对应的名称。比如：0代表狗，1代表猫，那么第一行就是dog，第二行就是cat。。下面会示例这个文件内容
+ backup:训练模型过程中产生的权重文件保存路径。有点像tensorflow的checkpoint路径

下面示例（`train`）train.txt所包含的内容：

```
./images/ee0fb09cfb52bde5157debcdca252fc7.jpg
./images/98e053938a2113363003ebd1bf9f81fe.jpg
./images/33104814567100bbd1034ef68d0bd39a.jpg
./images/115ed44ee5f896674210900491085839.jpg
```

### 1.3 obj.names

注意这个文件里面的内容顺序代表了标注文件中的分类名称：

 

### 1.4 weights权重文件

darknet可以基于其他预训练的权重文件再训练，重新训练时可能需要提供一个权重文件，可以比如[ImageNet](https://pjreddie.com/media/files/darknet19_448.conv.23)的预训练权重开始。



##  二 训练

训练脚本为：

```
./darknet detector train cfg/obj.data cfg/yolo-obj.cfg darknet19_448.conv.23
```

## 三 错误排查

### 3.1 error 1:

```
nvidia@tegra-ubuntu:~/workspace/cpp/darknet$ sudo ./darknet detector train ~/data/test_yolo_data/cfg/test.data ~/data/test_yolo_data/cfg/test.cfg ./weights/tiny-yolo-voc.weights 
[sudo] password for nvidia: 
test
First section must be [net] or [network]: No such file or directory
darknet: ./src/utils.c:253: error: Assertion `0' failed.
Aborted (core dumped)
```
将 darknet源代码 `cfg/voc-yolo.cfg`拷贝一份再修改参数。修改如下参数

+ 第三行修改为：

```
    batch=64
```

+ 第四行修改为,注意这个地方可能会导致 `could't open file train.txt`问题，可以尝试修改为其他，比如16,32

```
subdivisions=8
```

+ 第244行修改为

```
classes=4
```

+ 237行修改为。修改规则为 (classes+5)*5，当前有4个分类，所以是 (4+5)*5=45


```
filters=45
```


### 3.2 error 2

```
nvidia@tegra-ubuntu:~/workspace/cpp/darknet$ sudo ./darknet detector train ~/data/test_yolo_data/cfg/test.data ~/data/test_yolo_data/cfg/yolo-test.cfg ~/data/test_yolo_data/darknet19_448.conv.23 
yolo-test
layer     filters    size              input                output
    0 conv     32  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  32
    1 max          2 x 2 / 2   416 x 416 x  32   ->   208 x 208 x  32
    2 conv     64  3 x 3 / 1   208 x 208 x  32   ->   208 x 208 x  64
    3 max          2 x 2 / 2   208 x 208 x  64   ->   104 x 104 x  64
    4 conv    128  3 x 3 / 1   104 x 104 x  64   ->   104 x 104 x 128
    5 conv     64  1 x 1 / 1   104 x 104 x 128   ->   104 x 104 x  64
    6 conv    128  3 x 3 / 1   104 x 104 x  64   ->   104 x 104 x 128
    7 max          2 x 2 / 2   104 x 104 x 128   ->    52 x  52 x 128
    8 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256
    9 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128
   10 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256
   11 max          2 x 2 / 2    52 x  52 x 256   ->    26 x  26 x 256
   12 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512
   13 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256
   14 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512
   15 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256
   16 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512
   17 max          2 x 2 / 2    26 x  26 x 512   ->    13 x  13 x 512
   18 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024
   19 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512
   20 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024
   21 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512
   22 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024
   23 conv   1024  3 x 3 / 1    13 x  13 x1024   ->    13 x  13 x1024
   24 conv   1024  3 x 3 / 1    13 x  13 x1024   ->    13 x  13 x1024
   25 route  16
   26 conv     64  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x  64
   27 reorg              / 2    26 x  26 x  64   ->    13 x  13 x 256
   28 route  27 24
   29 conv   1024  3 x 3 / 1    13 x  13 x1280   ->    13 x  13 x1024
   30 conv    125  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 125
   31 detection
darknet: ./src/parser.c:281: parse_region: Assertion `l.outputs == params.inputs' failed.
Aborted (core dumped)

```

参考:https://groups.google.com/forum/#!topic/darknet/4_RNBWVEOnQ

解决方案就是修改上一步的 `filters=45`。

###  3.3 error 3

```
nvidia@tegra-ubuntu:~/workspace/cpp/darknet$ sudo ./darknet detector train ~/data/test_yolo_data/cfg/test.data ~/data/test_yolo_data/cfg/yolo-test.cfg ~/data/test_yolo_data/darknet19_448.conv.23 
yolo-test
layer     filters    size              input                output
    0 conv     32  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  32
    1 max          2 x 2 / 2   416 x 416 x  32   ->   208 x 208 x  32
    2 conv     64  3 x 3 / 1   208 x 208 x  32   ->   208 x 208 x  64
    3 max          2 x 2 / 2   208 x 208 x  64   ->   104 x 104 x  64
    4 conv    128  3 x 3 / 1   104 x 104 x  64   ->   104 x 104 x 128
    5 conv     64  1 x 1 / 1   104 x 104 x 128   ->   104 x 104 x  64
    6 conv    128  3 x 3 / 1   104 x 104 x  64   ->   104 x 104 x 128
    7 max          2 x 2 / 2   104 x 104 x 128   ->    52 x  52 x 128
    8 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256
    9 conv    128  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 128
   10 conv    256  3 x 3 / 1    52 x  52 x 128   ->    52 x  52 x 256
   11 max          2 x 2 / 2    52 x  52 x 256   ->    26 x  26 x 256
   12 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512
   13 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256
   14 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512
   15 conv    256  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 256
   16 conv    512  3 x 3 / 1    26 x  26 x 256   ->    26 x  26 x 512
   17 max          2 x 2 / 2    26 x  26 x 512   ->    13 x  13 x 512
   18 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024
   19 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512
   20 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024
   21 conv    512  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 512
   22 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024
   23 conv   1024  3 x 3 / 1    13 x  13 x1024   ->    13 x  13 x1024
   24 conv   1024  3 x 3 / 1    13 x  13 x1024   ->    13 x  13 x1024
   25 route  16
   26 conv     64  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x  64
   27 reorg              / 2    26 x  26 x  64   ->    13 x  13 x 256
   28 route  27 24
   29 conv   1024  3 x 3 / 1    13 x  13 x1280   ->    13 x  13 x1024
   30 conv     45  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x  45
   31 detection
mask_scale: Using default '1.000000'
Loading weights from /home/nvidia/data/test_yolo_data/darknet19_448.conv.23...Done!
Learning Rate: 0.001, Momentum: 0.9, Decay: 0.0005
Couldn't open file: train.txt
```

参考： https://groups.google.com/forum/#!msg/darknet/7JgHFfTyFHM/kPzfynNnAQAJ
这个解决办法是将 `test.cfg`文件中的 `subdivisions=8`修改为 `subdivisions=16`或者其他32,64等。但是这个解决办法对我无效，我后来发现需要在linux下重新 编辑一个新的文件`test.data`(voc.data)。是由于之前的文件是在windows下生成的，与ubuntu系统的编码格式不同。



参考 

https://timebutt.github.io/static/how-to-train-yolov2-to-detect-custom-objects/

https://pjreddie.com/darknet/yolo/







