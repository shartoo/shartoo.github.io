---
layout: post
title: 谷歌TensorFlow基本概念
description: 深度学习
category: blog
---

# start up

## 1.1 谷歌深度学习工具历史:
1. 第一代：**DistBelief** 由 Dean于2011年发起，主要产品有：
     + Inception (图像识别领域)
     + 谷歌Search
     + 谷歌翻译
     + 谷歌照片
2. 第二代：**TensorFlow** 由Dean于2015年11月发起，大部分DistBelief都转向了TensorFlow

## 1.2 产品特性


 |  概念 |描述|
 |---|----|       
 |编程模型|类数据流的模型|                      
 |语言|Python C++|                
 |部署|code once,run ererywhere|                
 |计算资源|cpu,gpu|                 
 |分布式处理|本地实现，分布式实现|                   
 |数学表达式|数学图表达式，自动分化|                    
 |优化|自动消除，kernel 优化，通信优化，支持模式，数据并行|           


## 1.3 计算图

```
import tensorflow as tf
b = tf.Variable(tf.zeros([100]))                   # 100维的向量，都初始化为0
w = tf.Variable(tf.random_uniform([784,100],-1,1)) # 784x100的矩阵
x = tf.placeholder(name="x")                       # 输入的占位符placeholder
relu = tf.nn.relu(tf.matmul(w,x)+b)                # Relu(Wx+b)
C =[...]                                           # 使用relu的一个函数计算代价

```
对应的计算图如下:
![计算图](/images/blog/tensorflow_basicconcept.png)

## 1.4 Tensorflow的代码样例

1. 构建数据流图的第一部分代码

```
import tensorflow as tf
import numpy as np
# 创建100个numpy的 x,y 假数据点，y = x*0.1+0.3
x_data = np.random.rand(100).astype("float32")
y_data = x_data*0.1+0.3
# 找出计算 y_data =W*x_data+b的w和b的值，虽然我们知道w=0.1,b=0.3,但是tensorflow会找到并计算出来
w = tf.Variable(tf.random_uniform)
```

# 2 tensorflow概览
要使用tensorflow的话，你需要理解以下概念:
+ 图代表了计算
+ 图需要在会话(Sessions)中执行
+ 张量(tensor)代表数据
+ 使用Variables来持有状态
+ 使用**feeds** 和 **fetches**来获得任何操作的输入输出数据

tensorflow的概览
+ 一个将计算转化为图的编程系统
+ 图中的节点是：
   + 操作(op):执行某些计算
   + 输入(input):一个或多个张量(tensorflow)
   + Tensor张量：一个有类型的多维数组

# 3 两个计算阶段
## 3.1 在图中计算
+ 图必须在Session中运行
+ 会话(Session)
    + 将图操作放入到设备上，比如CPUs和GPUs
    + 提供执行方法
    + 返回操作产生的张量，比如python中的**numpy ndarray对象**，以及C和C++**tensorflow::Tensor**实例。

## 3.2 图中的两个计算阶段
1. 构建阶段
    + 形成图
    + 创建图来代表神经网络并训练这个神经网络

2. 执行阶段
   + 使用会话执行途中的操作
   + 重复执行图中训练操作集合

3. 构建图
+ 开始那些不需要任何输入(source ops)的操作(op)，常量
+ 将它们的输出传入到其他做计算的操作
+ 操作构建者返回对象
    + 代表了结构化操作的输出
    + 将这些输出传入其他操作构建者作为输入

4. 默认图

 将节点加入此图的操作构建者

```
import tensorflow as tf
# 创建一个产生1x2的矩阵的常量操作，操作被作为节点加入到默认图
# 构建者的返回值代表了常量操作的输出
matrix1 = tf.constant([[3,3.]])
# 创建另外一个产生 2x1矩阵的常量操作
matrix2 = tf.constant([[2.0],[2.]])
```
   有三个节点：两个**constant**操作(ops)以及一个**matmul**操作
```
# 创建一个Matmul操作，将 matrix1和matrix2作为输入
# 返回值，‘product’，代表了矩阵相乘的结果
product = tf.matmul(matrix1,matrix2)
```

5. 在会话Session中运行图

+ 创建一个Session对象：应该在被关闭以释放资源
+ 没有参数，session构建者运行默认图

```
# 运行默认图
sess = tf.Session()
# 要运行matmul操作，我们调用了session的‘run()’方法，传入'producr'代表了matmul操作的输出。这即回调了matmul操作的输出结果
# 操作的输出以一个numpy的'ndarray'对象返回'result'
result = sess.run(product)
print result
# ==>[[12.]]
#关闭会话
sess.close()

```
6. Session运行图，Session.run()方法执行操作
7. 一个Session块(block)
    + 在块的结尾自动关闭
```
with tf.Session() as sess:
    result = sess.run([product])
    print result
```
8. GPU的使用
+ 将图定义转换为分布在各种计算资源，比如CPU和GPU之间的可执行操作
+ 如果有GPU，tensorflow会有限使用GPU

#4 交互使用

+ 在python环境中，比如Ipython,**InteractiveSession**类会被使用
+ **Tensor.eval()**和**Operation.run()**
+ 这可以避免必须用一个变量来保持一个session

```
# 进入一个交互的Tensorflow Session
import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.Variable([1.0,2.0])
y = tf.constant([3.0,3.0])
#使用'x'的initializer的 run() 方法初始化
x.initializer.run()

# 添加一个操作从'x'中抽取'a'，执行并打印结果
sub = tf.sub(x,a)
print sub.eval()
# ==>[-2,-1.]

# 关闭session
sess.close()
```

# 5 张量(Tensors)

+ Tensor(张量)数据结构代表了所有数据
+ 在计算图中只有张量在操作之间传递
+ n维数组或者列表
   + 静态类型，秩，或者 shape

**rank(秩)**

|rank|数学实体|python示例|
|---|---|---|
|0|Scalar(大小)|s =483|
|1|Vector(大小和方向)|v=[1.1,2.2,3.3]|
|2|Matrix(数据表)|m=[[1,2,3],[4,5,6],[7,8,9]]|
|3|3-Tensor(立方(cube)的数量)|t=[[[2],[4],[6],[8]],[[10],[12]]]|
|n|n-Tensor|同上|

**shape**

|Rank| Shape|维数|示例|
|---|---|---|
|0|[]|0-D|一个0-D张量，一个标量|
|1|[D0]|1-D|一个1-D张量，shape是[5]|
|2|[D0,D1]|2-D|一个2-D张量，shape是[3,4]|
|3|[D0,D1,D2]|3-D|一个3-D张量，shape[1,4,3]|
|n|[D0,D1,D2,...Dn]|n-D|一个n-D张量，shape是[D0,D1,...Dn]|

**数据类型**

|Data type|python类型|描述|
|---|---|---|
|DT_FLOAT|tf.float32|32位浮点类型|
|DT_DOUBLE|tf.float64|64位浮点类型|
|DT_INT64|tf.int64|64位有符号整型|
|DT_INT32|tf.int32|32位有符号整型|
|DT_INT16|tf.int16|16位有符号整型|
|DT_INt8|tf.int8|8位有符号整型|
|DT_UINT|tf.unit8|8位无符号整型|
|DT_STRING|tf.string|变量长度的字节数组，Tensor每个元素是一个字节数组|
|DT_BOOL|tf.bool|Boolean|
|DT_COMPLEX64|tf.complex64|由两个32位浮点数组成的复数，实数和大小部分|
|DT_QINT32|tf.qint32|量化操作中32位有符号整型|
|DT_QINT8|tf.qint8|量化操作中8位有符号整型|
|DT_QUINT8|tf.quint8|量化操作中8位无符号整型|

#6 变量
变量的创建、初始化、存储和载入
+ 为了持有和更新参数，在图中保持状态可以通过调用 **run()**方法
+ 内存buffer包含张量
+ 必须是明确初始化并且在训练期间和训练之后存储到磁盘上的
+ 类 **tf.Variable**
   + 构造器：变量的初始化值，一个任意类型和shape的张量
   + 构造之后，类型和shape都会固定
   + 使用**assign**操作op， validate_shape = False
## 6.1 创建
+ 传入一个张量作为初始值到 Variable构造方法中
+ 初始值：常量constants,序列化和随机值
   + tf.zeros(),tf.linspace(),tf.random_normal()
+ 固定shape：与操作的shape相同

```
# 创建两个变量
weight = tf.Variable(tf.random_normal([784,200],stddev=0.35),name ="weights")
biases = tf.Variable(tf.zeros([200]),name ="biases")

```
+ 调用 tf.Variable() 加入操作到图中

## 6.2 初始化
+ 添加一个操作并执行
+ tf.initialize_all_variables()

```
# 添加一个操作来初始化变量
init_op = tf.initialize_all_variables()
# 过后，执行model
with tf.Session() as sess:
    #运行初始化操作
    sess.run（init_op）
```
## 6.3 存储和恢复

+ **tf.saver**
+ 检查点文件：Variables都存储在二进制文件中，该文件包含了一个变量名到张量值得map

```
# 创建一些变量
v1 = tf.Variables(...,name ="v1")
v2 = tf.Variables(...,name="v2")
...
#添加一个操作来初始化变量
init_op = tf.initialize_all_variables()

# 添加操作来保存和恢复所有变量
saver = tf.train.Saver()
# 然后，运行模型，初始化变量，做一些操作，保存变量到磁盘中
with tf.Session() as sess:
     sess.run(init_op)
     # 对模型做一些操作
    .....
    #存储变量到磁盘中
    save_path = saver.save(sess,"/tmp/model.ckpt")
    print ("Model saved in file: %s"%save_path)

```
** 恢复**

```
with tf.Session() as sess:
    # 从磁盘中恢复变量
    saver.restore(sess,"/tmp/model.ckpt")
    print ("Model restored")
    # 做一些操作
```

## 6.4 选择哪些变量来存储和恢复

+ 在 ** tf.train.Saver()**中没有参数
    + 处理图中所有变量，每个变量都会被保存在该名字之下

+ 存储和恢复变量的子集
     + 训练5层神经网络，想训练一个新的6层神经网络，从5层圣经网络中恢复参数

+ 向**tf.train.Saver()**构造方法中传入一个Python词典:keys

```
# 创建一些变量
v1 = tf.Variables(...,name ="v1")
v2 = tf.Variables(...,name ="v2")
# 添加操作存储和恢复变量 v2,使用名字 "my_v2"
saver = tf.train.Saver({"my_v2":v2})
# 使用saver对象
...
```
## 6.5 简单计数器的示例代码

```
# 创建一个变量，初始化为标量0
state = tf.Variables(0,name="Counter")
# 创建一个操作来给"state"加1
one = tf.constant(1)
new_value = tf.add(state,one)
update = tf.assign(state,new_value)

# 在图被运行，变量必须是通过运行一个"init"操作被初始化。
# 我们首先要将"init"操作加入到图中
init_op = tf.initialize_all_variables()

# 运行图，和操作
with tf.Session() as sess:
    #运行 'init'操作
    sess.run(init_op)
    # 打印'state'的初始化值
    print (sess.run(state))
    # 运行更新'state'的操作，并打印'state'
    for _ in range(3):
        sess.run(update)
        print (sess.run(state))
# 输出
#0
#1
#2
#3
```

## 6.6 取数据Fetches
+ 在Session对象中调用**run()**方法来执行图，并传入张量来取回数据

```
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2,input3)
mul = tf.mul(input1,intermed)

with tf.Session() as sess:
     result = sess.run([mul,intermed])
     print (result)
# 输出
# [array([21.],dtype = float32),array([7.],dtype = float32)]

```

## 6.7 Feeds
+ 直接打包一个张量到图中的任何操作
+ 使用一个张量值临时替换一个操作的输出值
+ feed数据作为**run()**方法的一个参数
+ 仅仅用来运行调用被传入值
+ **tf.placeholder()**

```
input1 = tf.placeholder(tf.float32)
input2 =tf.placeholder(tf.float32)
output = tf.mul(input1,input2)

with tf.Session() as sess:
    print (sess.run([output],feed_dict = {input1:[7.],input2:[2.]}))

#输出
#[array([14.],dtype=float32)]

```

# 7 操作

|类别|示例|
|---|----|
|逐元素数学运算|Add,Sub,Mul,Div,Exp,Log,Greater,Less,Equal...|
|数组操作|Concat,Slice,Split,Constant,Rank,Shape,Shuffle..|
|矩阵运算|MatMul,MatrixInverse,MatrixDeterminant...|
|状态操作|Variable,Assign,AssignAdd...|
|神经元构建块|SoftMax,Sigmoid,ReLU,Convolution2D,MaxPool...|
|检查点操作|Save,Restore|
|队列和同步操作|Enqueue,Dequeue,MutexAcquire,MutexRelease,...|
|控制流操作|Merge,Switch,Enter,Leave,NextIteration|



