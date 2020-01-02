---
layout: post
title: pyspark机器学习库使用
description: 大数据
category: 大数据
mathjax: true
---

## 示例：垃圾邮件分类器

 以下代码使用两个 MLlib算法，HashingTF（从文本中构建词频特征向量的）和 LogisticRegressionWithSGD（使用随机梯度下降法来执行逻辑回归的算法）。

## 数据：

  spam.txt和normal.txt。都包含了垃圾邮箱和非垃圾邮箱，每行一封邮箱。将两篇文档转换为词频向量模型，然后训练逻辑回归模型来区分垃圾和非垃圾邮箱。

```
# -*- coding:UTF-8 -*-
#以下代码使用两个 MLlib算法，HashingTF（从文本中构建词频特征向量的）和 LogisticRegressionWithSGD（使用随机梯度
# 下降法来执行逻辑回归的算法）。
# 数据：
#spam.txt和normal.txt。都包含了垃圾邮箱和非垃圾邮箱，每行一封邮箱。将两篇文档转换为词频向量模型，
# 然后训练逻辑回归模型来区分垃圾和非垃圾邮箱。
 
import os
import sys
 
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark import SparkContext
from pyspark import SparkConf
 
# Path for spark source folder
os.environ['SPARK_HOME']="D:\javaPackages\spark-1.6.0-bin-hadoop2.6"
# Append pyspark  to Python Path
sys.path.append("D:\javaPackages\spark-1.6.0-bin-hadoop2.6\python")
 
if __name__ == "__main__":
    print "Program lanuch!"
    conf = SparkConf()
    conf.set("YARN_CONF_DIR ", "D:\javaPackages\hadoop_conf_dir\yarn-conf")
    conf.set("spark.driver.memory", "1gb")
    conf.setMaster("local")
    conf.setAppName("First_Remote_Spark_Program")
    sc = SparkContext(conf=conf)
 
    spam = sc.textFile("/home/xiatao/spam.txt")
    normal = sc.textFile("/home/xiatao/normal.txt")
 
    print "读取文件结束了"
    #创建一个 HashingTF实例将邮件文本映射到包含了10000个features的向量
    tf = HashingTF(numFeatures=10000)
    # 将每封邮件都切成单词，每个词映射到一个 features
    spamFeatures = spam.map(lambda email:tf.transform(email.split(" ")))
    normalFeatures = spam.map(lambda email: tf.transform(email.split(" ")))
    #分别给 正特征（垃圾邮件）和负特征（非垃圾邮件）创建 LabelPoint数据集
    positiveExamples = spamFeatures.map(lambda features:LabeledPoint(1,features))
    negativeExamples = normalFeatures . map(lambda features:LabeledPoint(0,features))
    trainingData = positiveExamples.union(negativeExamples)
    # 由于逻辑回归是个迭代算法，，最好缓存下
    trainingData.cache()
    # 使用SGD算法 运行逻辑回归
    print trainingData
    print "逻辑回归之前"
    model = LogisticRegressionWithSGD.train(trainingData)
    print "使用逻辑回归算法之前"
    #  测试一个 正特征数据 和 负特征数据，我们首先 应用HashingTF特征转换来获得向量，然后应用到模型中
    posTest = tf.transform("O M G GET cheap stuff by sending money to ....".split(" "))
    negTest = tf .transform("Hi Dad,i am studing Spark now...".split(" "))
    print "预测结果是:%g"%model.predict(posTest)
    print "预测结果是:%g"%model.predict(negTest)
    print "======================end==============="
```

## 数据类型

MLlib包含了一些特殊的数据类型，位于 `org.apache.spark.mllib.package(Java 或者Scala)` 或者 `pyspark.mllib(Python)`

+ **向量：**     一种数学向量，Spark支持稠密向量（每个位置都存储了值）和稀疏向量(只存储了非0值) 。可以通过 `mllib.linalg.Vector`类来创建向量

+ **LabeledPoint：** 一个标签化的数据点用在监督学习的算法中，比如分类和回归算法。包括一个特征向量和标签（值类型时float）位于 `mllib.regression`包里面

+ **Rating：** 用户产品评分，在 `mllib.recommendation`包中，用于产品推荐

+ **各种Model类：**每个Model都是一个训练算法的结果，并且基本上都有一个 predict()方法用来将模型应用新的数据点或者新数据点的RDD
大部分算法可以直接在 向量、LabeledPoint或者Rating的RDD上运行。

## 使用向量

首先：向量分两种，稀疏和稠密。对于10%左右元素非零的向量，推荐使用稀疏向量。既节省存储空间又提升速度。

其次：不同的语言构建向量时不同，在python可以简单的传入一个 NumPy数组到MLlib中创建一个稠密向量，或者使用`pyspark.mllib.linalg.Vectors`类来创建其他类型的向量。以下是python代码示例：

```
from numpy import array
from pyspark.mllib.linalg import Vectors
 
#创建一个稠密向量 <1.0,2.0,3.0>
# numpy可以直接传入到MLlib
denseVec1 = array([1.0,2.0,3.0]) 
#或者使用 Vectors类
denseVec2 = Vectors.dense([1.0,2.0,3.0])
#创建稀疏向量 <1.0,0.0,2.0,0.0>,其中(4)为向量元素个数，其他的是非零元素位置
#可以传入词典类型，也可以使用两个列表，分别是位置和值
sparseVec1 = Vectors.sparse(4,{0:1.0,2:2.0})
sparseVec2 = Vectors.sparse(4,[0,2],[1.0,2.0])
```

## 算法

如何调用和配置算法

## 特征抽取

`mllib.feature` 包包含 了几个常用的特征转换类，其中有将文本转换为特征向量的算法以及规划化和尺度的路径。

## TF-IDF

词频-逆向文档模型，是从文本中生成特征向量的最简单的办法。MLlib有两个计算 TF-IDF的算法：HashingTF和IDF都在mllib.feature 包中。HashingTF从文本中根据给定大小计算出词频向量。为了将词频映射到向量序位，HashingTF将每个单词对向量大小取模的哈希码，因而每个单词都会被映射到 0到 (size-1)(向量大小)。尽管多个词可能会被映射到相同的哈希码。MLlib开发者建议的向量大小为 2^18到2^20。

在python中使用 HashingTF

```
from pyspark.mllib.feature import HashingTF
 
sentence = "Hello world,Hello"
words = sentence.split(" ") #将语句切成词项列表
tf = HashingTF(10000)       #创建大小为10000的向量
tf.transform(words)
```

输出结果为:

```
SparseVector(10000,{3065:1.0,6861:2.0})
```

## 将整个RDD转换

```
rdd = sc.wholeTextFiles("data").map(lambda (name,text):text.split())
tfVectors = tf.transform(rdd)   #转换整个RDD
```
一旦创建了词频向量，就可以使用 IDF来计算逆向文档词频，然后乘以词频来计算TF-IDF。首先在一个 IDF对象上使用 fit()来获得 IDFModel，该模型代表了语料库中的逆向文档频率，然后调用 transform()来转换 TF向量为一个 IDF向量。

**在python中使用 TF-IDF**

```
from pyspark.mllib.feature import HashingTF,IDF
 
# 读取一些文档作为 TF向量
rdd = sc.wholeTextFiles("data").map(lambda (name,text):text.split(" "))
tf = HashingTF()
tfVectors = tf.transform(rdd).cache()
 
#计算 IDF，然后计算 TF-IDF
idf = IDF()
idfModel = idf.fit(tfVectors)
tfidfVectors = idfModel.transform(tfVectors)
```

## Scaling

大部分机器学习算法会考虑特征向量中的每个元素的大小(尺度)，因而当特征都均衡时（比如都在范围 0-1之间）时算法表现最好。一旦建立好特征向量，可以使用 MLlib中的 StandardScaler类来解决尺度问题。先创建一个 StandardScaler，然后在数据集上调用 fit() 方法来获得一个 StandardScalerModel，然后在模型上调用 transform() 来均衡(尺度平衡)数据集。

```
from pyspark.mllib.feature import StandardScaler
 
vectors = [Vectors.dense([-2.0,5.0,1.0]),Vectors.dense([2.0,0.0,1.0])]
dataset = sc.parallize(vectors)
scaler =  StandardScaler(withMean = True,withStd = True)
model = scaler.fit(dataset)
result =model.transform(dataset)
```

## 规范化

Normalizer类允许用户将向量规范化到长度为1的空间内，使用 `Normalizer().transform(rdd)`即可。默认情况下是将数据按照欧几里得距离规范化，可以向Normalizer()中传入参数改变，如果传入的是3，将会被规范化到 L^3的空间上。

## 统计

 Spark提供了一些直接应用到RDD上的统计函数，位于`mllib.stat.Statistics`类。


+ **Statistics.colStats(rdd) :**计算一个RDD向量的统计概要，保存向量集合每一列的最小值、最大值、平均值以及方差。

+ **Statistics.corr(rdd,method)：**计算RDD向量列之间的相关性，使用Pearson 或者Spearman（方法必须是这两者中的一个）

+ **Statistics.corr(rdd1,rdd2,method)：**计算两个RDD向量浮点值之间的相关性。method同上

+ **Statistics.chiSqTest(rdd)：**计算有label标签的LabeledPoint对象的RDD的每个特征的皮埃尔独立性检测。

##分类和回归

分类和回归两个常见的监督学习形式，算法尝试从打过标签的训练数据对象中预测变量。不同之处在于预测变量的类型：分类中所有分类是限定（离散）的，回归中变量预测是连续的。
 在MLlib中分类和回归都是用 LabeledPoint类，也即“数据类型”。一个LabeledPoint由标签(一般是double，但是也可以被设置成离散的)和特征向量组成。

##线性回归

      线性回归是是回归算法中最简单的回归算法，预测特征的线性组合变量输出。MLlib支持Lasso回归和ridge回归。通过 `mllib.regression.LineRegressionWithSGD`,`LassonWithSGD`和`RidgeRegressionWithSGD`类可以使用，在MLlib中遵从一致的命名模式，当问题牵扯到多个算法时，类名中"With"部分所使用的算法。此处SGD即 Stochastic Gradient Descent(随机梯度下降)。这些类都有几个参数来调整算法:

+ **numIterations:**算法迭代次数，默认是100
+ **stepSize：** 梯度下降步长（默认是1.0）
+ **intercept:** (截距)是否向数据中加入截距或者 偏置特征，也即特征值始终为1的。默认是不添加的
+ **regPram：**Lasso和ridge回归的正则参数

python中线性回归算法示例:

```
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import LinearRegressionWithSGD
 
points = #创建一个 LabeledPoint的RDD
model = LinearRegressionWithSGD.train(point,iterations = 200,intercept=True)
print "weight:%s,intercept: %s"%(model.weights,model.intercept)
```

## 逻辑回归

      逻辑回归是一个将数据样例分为正、负的二分类平面。在MLlib中使用LabeledPoint 标签0 和标签1并返回LogisticRegressionModel来预测新的数据点。
      
逻辑回归有着与线性回归十分相似的API，不同之处在于逻辑回归使用的算法时SGD和LBFGS。通常选择LBFGS。可以在`mllib.classification.LogisticRegressionWithLBFGS` 和`WithSGD`类中找到。
      
      这些逻辑回归算法中的`LogstisticRegressionModel`给每个点计算一个0到1之间的分值。然后给予给定的阈值返回0或1，可以通过设置 setThreshold来改变阈值，也可以通过 clearThreshold()方法清除阈值设置，清除之后 predict()将返回原始的分值。

## SVM支持向量机

   SVM也是一个返回线性分类平面的二分类方法，

+ **协同过滤和推荐算法** 

         协同过滤是一种根据用户与物品的交互评分数据来推荐新物品的技术。仅需要一张 用户/产品 交互清单即可：可以是确定交互（直接在网站上给产品评分）或者隐式交互（用户浏览了某个产品，但是没有评分）。根据这些，协同过滤就知道哪些产品之间有相似性，以及哪些用户之间存在相似。

+ **交替最小二乘法**

          产品和用户构成的M*N矩阵(产品有M个，用户有N个)，但这个矩阵是稀疏的，只有部分评分，ALS就是填满矩阵中缺失值得，填满的过程就是推荐过程。MLlib包含了一个ALS的实现，一个易于在集群中拓展的协同过滤算法，位于 mllib.recommendation.ALS
使用以下参数：

+ **rank：**特征向量秩大小，越大的秩会得到更好的模型，但是计算消耗也相应增加。默认是 10

+ **iteration：** 算法迭代次数（默认是10）

+ **lambda：**正则参数，默认是 0.01。详细解释参考 https://www.zhihu.com/question/31509438

+ **alpha：**在隐式ALS中用于计算置信度的常量，默认为1.0

+ **numUserBlocks,numProductBlocks：**将用户和产品数据分解的块数目，用来控制并行度；你可以传入-1来让MLlib自动决定。

要使用ALS，你需要给定一个 `mllib.recommendation.Rating`对象的RDD，每个都包含 用户ID，产品ID和评分。注意：每个ID都必须是是一个32位整型数据，如果你的ID是字符串或者比较大的数据，推荐使用哈希之后的数据。

ALS返回一个 `MatrixFactorizationModel`来代表结果，此结果可以用来给键值对RDD(userID,productID)使用predict()预测评分。另外，你可以使用 model.recommendProducts(userID,numProducts) 找到 top numProducts的产品给指定用户。切记，不像MLlib中其他模型，MatrixFactorizationModel是较大的，为每个用户和产品持有一个向量。这表明它不能存储在磁盘上然后再载入并用在另外部分代码，但是你可以存储它产生的特征向量RDD，比如`model.userFeatures`和`model.productFeatures`到分布式文件系统中。

最后，有两种类型的ALS：对于确定评分（默认，使用 ALS.train()）和隐式评分（使用 ALS.trainImplicit()）。对于确定评分，每个用户对产品的评分必须是分值（比如说1-5星），然后预测评分也是分值。对于隐式评分，评分代表了用户与给定产品项的交互置信度，然后预测项也是置信度。
