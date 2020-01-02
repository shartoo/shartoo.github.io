---
layout:     post
title:      大数据：spark mllib python使用示例
category: 大数据
description: 大数据
mathjax: true
---

## 机器学习的背景知识

 > 监督学习的中点是** 在规则化参数的同时最小化误差**。最小化误差是为了让我们的模型拟合我们的训练数据，而规则化参数是防止我们的模型过分拟合我们的训练数据。参数太多，会导致我们的模型复杂度上升，容易过拟合，也就是我们的训练误差会很小。但**训练误差**小并不是我们的最终目标，我们的目标是希望模型的**测试误差**小，也就是能准确的预测新的样本。所以，我们需要保证模型“简单”的基础上最小化训练误差，这样得到的参数才具有好的泛化性能（也就是测试误差也小），而模型“简单”就是通过规则函数来实现的。另外，规则项的使用还可以约束我们的模型的特性。这样就可以将人对这个模型的先验知识融入到模型的学习当中，强行地让学习到的模型具有人想要的特性，例如稀疏、低秩、平滑等等。要知道，有时候人的先验是非常重要的。


来源于：   http://blog.csdn.net/zouxy09/article/details/24971995

#  线性模型

## 一  数学公式


&emsp; &emsp; 许多机器学习方法都可以被转换为一个凸函数优化问题，比如查找凸函数f（自变量是w，在代码中称为权重，自变量有d维）最小值。通常，我们可以将这些写成 $ min_{w\epsilon R^d}f(w) $ ，其目标函数是以下形式
$$  f(w) := \lambda\, R(w) +\ frac1n \sum_{i=1}^n L(w;x_i,y_i) \label{eq:regPrimal}$$

   
&emsp;
&emsp; 
此处向量$x_{i}\epsilon R^d$是训练数据，对于$1\leq i\leq n$ 和 $y_{i}\epsilon R$是我们需要预测的标签。如果$L(w;x,y)$可以被表示为 $W^T x$和$y$的函数，则可以调用 ***linear***方法。
   
&emsp;
&emsp; 
目标函数***f***分为两部分：控制模型复杂度的正则化部分，模型在训练数据集上误差评估的损失度量部分。损失度量函数$L(w;.)$是一个在域$w$上的凸函数。固定的正则化参数$\lambda \geq 0$(代码中是参数***regParam***)定义了权衡最小化损失（比如训练误差）和最小化模型复杂度（比如，防止过拟合）之间的平衡。

## 二 误差函数    

下表概括了损失函数和它们在spark.mllib支持的的梯度和分梯度方法. 

|损失   |loss function $L(w;x,y)$$\;\;\;\;\;\;\;\;$|梯度或分梯度|
|---|--------------------|----------|
|hinge loss(SVM)|$max${0,$1-yw^{T}x$},$y \epsilon$ {-1,+1} | $ -y\cdot x (if\, yw^Tx < 1)\;\; 0 (otherwise))$|
|logstic loss(逻辑回归)|$log(1+exp(-yw^Tx)),y\epsilon {-1,+1}$ | $-y(1-\frac {1}{1+exp(-yw^Tx)})\cdot x$|
|squared loss(最小二乘)|$\frac {1}{2}(w^Tx-y)^2, y\epsilon R$ | $(w^Tx-y)\cdot x$|

## 三 规则化

  &emsp;&emsp;规则化的目的是简化模型并避免过拟合，规则化函数Ω(w)也有很多种选择，一般是模型复杂度的单调递增函数，模型越复杂，规则化值就越大。比如，规则化项可以是模型参数向量的范数。然而，不同的选择对参数w的约束不同，取得的效果也不同
&emsp;&emsp;关于L1范式和L2范式:
+ L0 范式：L0范数是指向量中非0的元素的个数。如果我们用L0范数来规则化一个参数矩阵W的话，就是希望W的大部分元素都是0，换句话说，让参数W是稀疏的。
+ L1 范式:  L1范数是指向量中各个元素绝对值之和，也称叫“稀疏规则算子”（Lasso regularization）。
既然L0可以实现稀疏，为什么不用L0，而要用L1呢？一是因为L0范数很难优化求解（NP难问题），二是L1范数是L0范数的最优凸近似，而且它比L0范数要容易优化求解。
+ L2范式：
在回归里面，有人把有它的回归叫“岭回归”（Ridge Regression），有人也叫它“权值衰减weight decay”。它的作用是改善过拟合。
L2范数是指向量各元素的平方和然后求平方根。我们让L2范数的规则项||W||2最小，可以使得W的每个元素都很小，都接近于0，但与L1范数不同，它不会让它等于0，而是接近于0。而越小的参数说明模型越简单，越简单的模型则越不容易产生过拟合现象。
&emsp;&emsp;目前在 spark.mllib中支持的正则化如下：
|范式|      regularizer $R(w)$      |            梯度或子梯度       |
|----|-----------|-------|
|zero(unregularized)|0|0|
|L2|$\frac{1}{2}$\|\|w\|\|$_2^2$|w|
|L1|$\|\|w\|\|_1$|$sign(w)$|
|elastic net|$\alpha\|\|w\|\|_1+(1-\alpha)\frac{1}{2}\|\|w\|\|_2^2$|$\alpha sign(w)+(1-\alpha)w$|

*此处$sign(w)$是由项$w$中所有$sign(-1,+1)$组成的向量*

## 三 优化

&emsp;&emsp;线性方法使用凸函数来优化目标函数. spark.mllib使用两个方法，SGD和LBFGS（Limited-Memory Quasi-Newton Method）。当前，大多数算法API都支持Stochastic Gradient Descent（随机梯度下降），和少部分支持LBFGS。

## 四  分类

&emsp;&emsp;分类旨在将数据项切分到不同类别。***spark.mllib***提供了两个线性分类方法：线性SVM和逻辑回归。线性SVM只支持二分类，逻辑回归既支持二分类也支持多分类。这两种方法，***spark.mllib***都支持L1和L2范式规则化。在**MLlib**中训练数据集合以 *LabeledPoint*类型的RDD代表，其中label（标签）是从0开始0,1,2...的类别索引。
&emsp;&emsp;**注意**：指导手册中的，二分类标签$y$要么是 1 要么是 -1，这是为了方便在公式里，但是在***spark.mllib**里面是以0代表公式中的-1的

## 五 线性SVM  

### 5.1 线性SVM概要 

&emsp;&emsp线性SVM的误差函数是由hingle loss给出:$$L(w;x,y) :=max\lbrace0,1-yw^Tx\rbrace$$
线性SVM默认使用L2范式规则化训练数据，同时是支持L1范式规则的，此时就变成一个线性问题。
线性SVM的输出是一个SVM模型。对于一个新数据点，以$x$表示，模型将会基于$w^Tx$的值预测。默认情况下，如果$w^Tx\geq 0$那么输出为1，否则输出为0。

### 5.2 示例代码

&emsp;&emsp;一下代码展示了如何载入数据，创建SVM模型，根据模型预测并计算训练误差。

```
# -*- coding:utf-8 -*-
"""
学习使用spark.mllib中 SVM模型。代码展示了如何载入数据，创建SVM模型，根据模型预测并计算训练误差。
"""

from pyspark.mllib.classification import SVMWithSGD,SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext
from pyspark import SparkConf

import os
import sys
import logging
# Path for spark source folder
os.environ['SPARK_HOME']="D:\javaPackages\spark-1.6.0-bin-hadoop2.6"
# Append pyspark  to Python Path
sys.path.append("D:\javaPackages\spark-1.6.0-bin-hadoop2.6\python")
sys.path.append("D:\javaPackages\spark-1.6.0-bin-hadoop2.6\python\lib\py4j-0.9-src.zip")

conf = SparkConf()
conf.set("YARN_CONF_DIR ", "D:\javaPackages\hadoop_conf_dir\yarn-conf")
conf.set("spark.driver.memory", "2g")
conf.setMaster("yarn-client")
conf.setAppName("TestSVM")
logger = logging.getLogger('pyspark')
sc = SparkContext(conf=conf)
mylog = []
#载入数据并解析
def parsePoint(line):
  values = [float(x) for x in line.split(" ")]
    return LabeledPoint(values[0],values[1:])
data = sc.textFile("/home/xiatao/machine_learing/SVM/sample_svm_data.txt")

parseData = data.map(parsePoint)
#创建SVM模型
model = SVMWithSGD.train(parseData,iterations=100)
# 评估模型
labelsAndPoints = parseData.map(lambda p:(p.label,model.predict(p.features)))
trainError = labelsAndPoints.filter(lambda (v,p):v!=p).count()/float(parseData.count())
mylog.append("SVM模型测试，训练误差是:")
mylog.append(str(trainError))
sc.parallelize(mylog).saveAsTextFile("/home/xiatao/machine_learing/SVM/log/")
#存储和载入模型
model.save(sc,"/home/xiatao/machine_learing/SVM/SVMModelSave")
sameModel = SVMModel.load(sc,"/home/xiatao/machine_learing/SVM/SVMModelSave")
```

## 六 逻辑回归

### 6.1 逻辑回归概要

&emsp;&emsp;逻辑回归在二分类预测中广泛使用，其误差函数是下式 logistic loss:$$L(w;x,y):=log(1+\exp(-yw^Tx))$$
对于二分类问题，算法的输出结果是一个二项式逻辑回归模型。对于给定新数据点，以$x$表示，使用logistic函数来预测:$$f(z) =\frac{1}{1+e^{-z}}$$。其中$z =w^Tx$，如果$f(w^Tx)>0.5$输出为正，否则为负。与线性SVM不同的是，逻辑回归模型的原始输出有一个概率解释（即，x是正的概率）。
&emsp;&emsp;二项式逻辑回归可以生成多项式逻辑回归并用来训练和预测多分类问题。比如说，对于**K**个可能的输出结果，其中一个可以选定为轴，其余**k-1**则与此轴对立。在spark.mllib中第一个被选中的类0就是轴类。
&emsp;&emsp;对于多分类问题，算法将会输出一个多项式逻辑回归模型，包含了**k-1**个与第一个类对立的二项式逻辑回归模型。对于新数据点，**k-1**个模型将会运行，其中有最大概率的模型即预测的模型。
&emsp;&emsp;spark中实现了两个算法来解决逻辑回归问题：mini-batch gradient（梯度下降）和L-BFGS。参考[batch-GD， SGD， Mini-batch-GD， Stochastic GD， Online-GD区别](http://www.bubuko.com/infodetail-898846.html)  spark推荐L-BFGS梯度下降以获得更快的收敛。

### 6.2 逻辑回归代码    
```
# -*- coding:utf-8 -*-

"""
测试逻辑回归代码
"""

from pyspark.mllib.classification import  LogisticRegressionWithSGD,LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext
from pyspark import SparkConf

import os
import sys
import logging

# Path for spark source folder
os.environ['SPARK_HOME']="D:\javaPackages\spark-1.6.0-bin-hadoop2.6"
# Append pyspark  to Python Path
sys.path.append("D:\javaPackages\spark-1.6.0-bin-hadoop2.6\python")
sys.path.append("D:\javaPackages\spark-1.6.0-bin-hadoop2.6\python\lib\py4j-0.9-src.zip")

conf = SparkConf()
conf.set("YARN_CONF_DIR ", "D:\javaPackages\hadoop_conf_dir\yarn-conf")
conf.set("spark.driver.memory", "2g")
conf.setMaster("yarn-client")
conf.setAppName(
"TestLogisticRegression"
)

logger = logging.getLogger('pyspark')

sc = SparkContext(conf=conf)
mylog = []


#载入和解析数据
def parsePoint(line):

    
values = [float(x) for x in line.split(" ")]
return LabeledPoint(values[0],values[1:])

data = sc.textFile("/home/xiatao/machine_learing/logistic_regression/sample_svm_data.txt")
parseData  = data.map(parsePoint)
#创建模型
model = LogisticRegressionWithSGD.train(parseData)

#评估模型
labelaAndPoints = parseData.map(lambda p:(p.label,model.predict(p.features)))
trainError = labelaAndPoints.filter(lambda (k,v):k!=v).count()
/
float(parseData.count())
mylog.append("逻辑回归的误差是:")
mylog.append(trainError)

# 存储和载入模型
model.save(sc,"/home/xiatao/machine_learing/logistic_regression/logisticregression_model/")
sc.parallelize(mylog).saveAsTextFile("/home/xiatao/machine_learing/logistic_regression/log/")
logisticregression_model = LogisticRegressionModel.load(sc,"/home/xiatao/machine_learing/logistic_regression/logisticregression_model")

```
## 7 回归

&emsp;&emsp;**Linear least squares, Lasso, and ridge regression**
&emsp;&emsp;Linear least squares是回归问题最常用的公式，其误差函数如下：$$L(w;x,y):=\frac{1}{2}(w^Tx-y)^{2}$$
使用不同的规则参数将会派生出不同的相关回归方法；其中的 **ordinary least squares**和**linear least squares**不使用规则参数，ridge regression(岭回归)使用L2规则参数，Lasso使用L1规则参数。所有的这些模型，其平均误差或者训练误差$$\frac{1}{n}\sum_{i=1}^n(w^Tx-y_i)^2$$ 即均方差。
&emsp;&emsp;一下代码展示了如何载入数据、转换为LabeledPoint类型的RDD。然后使用 LinearRegressionWithSGD来创建简单线性模型来预测标签值。最后再计算均方差来评估适应度。

```
# -*- coding:utf-8 -*-
from pyspark.mllib.regression import LabeledPoint,LinearRegressionWithSGD,LinearRegressionModel
from pyspark import SparkContext
from pyspark import SparkConf

import os
import sys
import logging
# Path for spark source folder
os.environ['SPARK_HOME']="D:\javaPackages\spark-1.6.0-bin-hadoop2.6"
# Append pyspark  to Python Path
sys.path.append("D:\javaPackages\spark-1.6.0-bin-hadoop2.6\python")
sys.path.append("D:\javaPackages\spark-1.6.0-bin-hadoop2.6\python\lib\py4j-0.9-src.zip")

conf = SparkConf()
conf.set("YARN_CONF_DIR ", "D:\javaPackages\hadoop_conf_dir\yarn-conf")
conf.set("spark.driver.memory", "2g")
conf.setMaster("yarn-client")
conf.setAppName("TestSimpleLinearRegression")
logger = logging.getLogger('pyspark')
sc = SparkContext(conf=conf)

#载入数据
def parsePoint(line):
  values = [float(x) for x in line.replace(',',' ').split(' ')]
    return LabeledPoint(values[0],values[1:])

mylog = []
data = sc.textFile("/home/xiatao/machine_learing/linear_regression/data/lpsa.data")
parseData = data.map(parsePoint)
#创建模型
model = LinearRegressionWithSGD.train(parseData,iterations = 10,step=0.000001)
#评估模型误差
valuesAndPres = parseData.map(lambda p:(p.label,model.predict(p.features)))
MSE = valuesAndPres.map(lambda (v,p):(v-p)**2).reduce(lambda x,y:x+y)/valuesAndPres.count()
mylog.append("简单线性回归误差是：")
mylog.append(MSE)
sc.parallelize(mylog).saveAsTextFile("/home/xiatao/machine_learing/linear_regression/log")
#存储 和使用模型
model.save(sc,"/home/xiatao/machine_learing/linear_regression/SimpleLinearRegressionModel")
sameMode = LinearRegressionModel.load(sc,"/home/xiatao/machine_learing/linear_regression/SimpleLinearRegressionModel")</pre>
```

### 流线性回归
如果数据是以流的形式到达，在线适配回归模型、新数据到达时跟更新模型参数是很有用的。*spark.mllib*当前支持使用 **ordinary least squares**的线性回归。适应过程类似于离线使用，一批新数据到达时预测适应值，以此来不断地更新流中的新数据的回应值（回归值）。
&emsp;&emsp;如下代码演示了如何训练和测试来自两个不同文本格式的输入流，将流解析成labeled point,使用第一个流来拟合回归模型，并在第二个流中作预测。注意，当训练目录 *"/home/xiatao/machine_learing/streaming_linear_regression/data* 新增数据时，相应的预测目录*/home/xiatao/machine_learing/streaming_linear_regression/predict*就会产生相应的结果。

```
# -*- coding:utf-8 -*-
# 流线性回归模型测试
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import StreamingLinearRegressionWithSGD
from pyspark.streaming import StreamingContext
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext
from pyspark import SparkConf

import os
import sys
import logging
# Path for spark source folder
os.environ['SPARK_HOME']="D:\javaPackages\spark-1.6.0-bin-hadoop2.6"
# Append pyspark  to Python Path
sys.path.append("D:\javaPackages\spark-1.6.0-bin-hadoop2.6\python")
sys.path.append("D:\javaPackages\spark-1.6.0-bin-hadoop2.6\python\lib\py4j-0.9-src.zip")

conf = SparkConf()
conf.set("YARN_CONF_DIR ", "D:\javaPackages\hadoop_conf_dir\yarn-conf")
conf.set("spark.driver.memory", "2g")
conf.setMaster("yarn-client")
conf.setAppName("TestStreamLinearRegression")
logger = logging.getLogger('pyspark')
sc = SparkContext(conf=conf)
mylog = []

#第一步创建 StreamingContextssc = StreamingContext(sc,1)

#载入和解析数据
def parse(lp):
  label = float(lp[ lp.find('(')+1:lp.find(',')  ])
    vec = Vectors.dense(lp[lp.find('[')+1:lp.find(',')].split(','))
    return LabeledPoint(label,vec)
# 训练集和测试集的数据每行的格式为 (y,[x1,x2,x3]),其中y是标签，x1,x2,x3是特征。
# 训练集中数据更新时，测试集目录就会出现预测值。并且数据越多，预测越准确
trainingData = ssc.textFileStream("/home/xiatao/machine_learing/streaming_linear_regression/data").map(parse).cache()
testData = ssc.textFileStream("/home/xiatao/machine_learing/streaming_linear_regression/predict").map(parse)
#创建 权值为0的初始化模型
numFeatures= 3
model = StreamingLinearRegressionWithSGD()
model.setInitialWeights([0.0,0.0,0.0])
# 为训练流和测试流登记，并启动作业
model.trainOn(trainingData)
mylog.append(model.predictOnValues(testData.map(lambda lp:(lp.label,lp.features))))

ssc.start()
ssc.awaitTermination()</pre>
```
