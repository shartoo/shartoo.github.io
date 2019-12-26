---
layout:     post
title:      大数据：spark mllib决策树
category: 大数据
description: 大数据
---

# 一 基本算法

&emsp;&emsp;决策树是一个在特征空间递归执行二分类的贪心算法。决策树预测所有叶子节点分区的标签。为了在树的每个节点最大化信息增益，其每个分区都是基于贪心策略从可能分裂集合里选择一个最佳分裂(split)。也即，每个数节点分裂的选择是从集合 $argmaxIG(D,s)$，其中$IG(D,s)$是信息增益，而s是应用到数据集D上的分裂。

## 二 节点不纯度和信息增益

节点不纯度是用以衡量节点同质化标签的度量。当前为了分类提供两种不纯度方法（Gini impurity和信息熵），为回归提供了一个不纯度度量（方差）。


|impurity(不纯度)|作业|公式|描述|
|-------------|----|----|----|
|Gini impurity|分类| $\sum_{i=1}^cf_i(1-f_i)$ | $f_i$ 是某个节点上标签为i的频率,C是标签数据|
|信息熵|分类| $\sum_{i=1}^c-f_i\log(f_i)$|$f_i $是某个节点上标签为i的频率,C是标签数据|
|方差|回归| $\frac{1}{N}\sum_{i=1}^N(y_i-\mu)^2$|$y_i$是某个数据实例的标签，N是数据实例的总数，$\mu$是由$\frac{1}{N}\sum_{i=1}^Ny_i$均值|


信息增益是衡量父母节点的不纯度与两个孩子节点不纯度权值求和的差异。假设一个分裂$s$将数据集 D(包含N个元素)分裂成两个子集合 $D_{left}$（包含$N_{left}$个元素）和 $D_{right}$ (包含 $N_{right}$ )，相应的信息增益是:

$$

IG(D,s) = Impurity(D)-\frac{N_{left}}{N}Impurity(D_{left})-\frac{N_{right}}{N}Impurity(D_{right})

$$

## 三 分裂候选

### 3.1 连续特征

对于小数据集在单机上的实现，对每个连续特征来说其分裂候选一般是该特征的唯一值。有些实现会将特征值排序然后使用排序后的唯一值作为分裂候选以达到更快的计算速度。
&emsp;&emsp;对于大规模分布式数据来说排序的特征值是代价高昂的。通过对部分抽样数据进行位数计算来近似计算其分裂候选，以此来实现排序。排序后的分裂会创建“分箱”，可以通过参数***maxBins***来指定最大分箱数。
&emsp;&emsp;注意，分箱数目可以比数据实例数目大（这种情况比较少见，由于默认的***maxBins***是32）。如果分裂时条件不满足了，决策树会自动减少分箱数目。

### 3.2 分类特征

对于一个分类特征，有M个可能的取值（类别），可能会有$2^{M-1}-1$个分裂候选。对于二分类(0/1)和回归，我们可以通过对类别特征排序（用平均标签）将分裂候选减少至**M-1**。例如对于某个二分类问题，1个类别特征，3个分类A,B,C，相应的标签为1的比例为0.2,0.6,0.4，类别特征排序为A,C,B。两个分裂候选是A|C,B和A,C|B，其中竖线代表分裂。
&emsp;&emsp;在多分类中，所有的$2^{M-1}-1$个可能的分裂无论何时都可能会被使用到。如果$2^{M-1}-1$比参数***maxBins***大，使用一个与二分类和回归分析中类似的启发式方法。***M***个类别特征都是根据不纯度排序的。

## 四 停止规则

递归的构建树过程会在某个节点满足以下条件时停止：

1. 树深度已经等于训练参数***maxDepth***。

2. 分裂候选产生的信息增益都小于参数***minInfoGain***。

3. 分裂候选已经不能产生孩子节点，满足每个孩子节点有至少***minInstancePerNode***训练集实例。

## 五 参数设置问题

以下参数需要设置但不需要调节。

1. **algo**：分类还是回归。

2. **numClass**:分类的类别数目（只对分类）

2. **categoricalFeaturesInfo**：设置哪些特征是类别以及每个这些特征值可以取多少类别值。此参数以map的形式给出，所有不在这个map中的特征都会被视为连续的。map的取值示例如下:

+ Map(0->2,4->10....) 指明，特征0 是二分类（取值为0或1），特征4有10个类别（取值是0-9）

+ **注意**：你并不需要配置 *categoricalFeaturesInfo*。算法依然会运行并给出不错的结果，然而如果可特征化的值设计得很好，算法可以有更好的性能。

## 六 停止标准

&emsp;&emsp;这些参数决定算法何时停止（增加节点），调节以下参数时，注意在测试数据集上验证并避免过拟合。

+ **maxDepth**:树的最大深度。越深的树（可能会获取更高的准确率）计算代价越高，但是它们也更耗时同时更可能过拟合。

+ **minInstancesPerNode**:对于一个可能会进一步分裂的节点，它的子节点必须有至少这么多个训练实例数据。此参数一般和随机森林一起使用，因为这些会比单独的树要训练得更深。

+ **minInfoGain**:对于可能会进一步分裂的节点，分裂必须增加这么多信息增益。

## 七 调节参数

&emsp;&emsp;这些参数可以调节，但是注意在测试数据集上验证并避免过拟合。
+ **maxBins**:离散化连续型变量时使用的分箱数。增加 **maxBins**使得算法考虑更多的分裂候选并产生更细粒度的分裂决策，然而会增加计算消耗和组件间沟通成本。注意：对于任何可类别话的特征，参数**maxBins**必须至少是类别**M**最大值。
+ **maxMemoryInMB**:进行统计时使用的内存量。默认值保守取到256MB，足以使得决策树在大多数场景适用。增大此参数可以减少数据传输让训练过程更快。
实现细节：为了更快的处理速度，决策树算法收集每组会分裂的节点的统计数据（而不是一次一个节点）。能放入一个组中处理的节点是由内存需求决定的（不同的特征不同）。参数**maxMemoryInMB**配置了每个使用这些统计的worker的内存限制。
+ **subsamplingRate**:学习决策树的训练数据集比例。这个参数大多用在训练树的集合（随机森林、GradientBoostedTrees（渐变提振树））中，用以在袁术数据集中抽样数据。在单个决策树中，此参数并没有那么重要，因为训练数据并不是最大的限制。
+ **impurity**:在分裂候选中筛选衡量不纯度的参数，这个参数必须与**algo**参数相对应。

## 八 缓存和检查点

当参数**maxDepth**设置得很大时，有必要开启节点ID缓存和检查点。在随机森林中，如何参数**numTrees**设置得很大时，也比较有用。
+ **useNodeIdCache**：如何此参数设置为* ture*，算法将会避免在每次迭时传入当前模型（tree ,trees）。算法默认会让当前模型与executors交流，使得executors每个树节点能够达到训练实例要求。当开启此参数时，算法将会缓存这部分信息。

节点ID缓存会生成一些RDD（每次迭代时生成一个）。这种很长的lineage(血缘)会导致性能问题，但是检查点中间RDD可以缓和这些问题，**注意**只有当***useNodeIdCache***设置为***true***检查点才可用。
+ **checkpointDir**:节点ID缓存RDD的检查点目录。
+ **checkpointInteral**:节点ID缓存RDD的频率，设置的过小会导致过量的写入HDFS，设置得太大时会使得executors失败并需要重新计算时等待太长。

## 九 代码实例

以下代码展示了如何载入一个**LIBSVM**数据文件，解析成一个**LabeledPoint**RDD，然后使用决策树，使用Gini不纯度作为不纯度衡量指标，最大树深度是5.测试误差用来计算算法准确率。

```
# -*- coding:utf-8 -*-
"""
测试决策树
"""

import os
import sys
import logging
from pyspark.mllib.tree import DecisionTree,DecisionTreeModel
from pyspark.mllib.util import MLUtils

# Path for spark source folder
os.environ['SPARK_HOME']="D:\javaPackages\spark-1.6.0-bin-hadoop2.6"
# Append pyspark  to Python Path
sys.path.append("D:\javaPackages\spark-1.6.0-bin-hadoop2.6\python")
sys.path.append("D:\javaPackages\spark-1.6.0-bin-hadoop2.6\python\lib\py4j-0.9-src.zip")

from pyspark import SparkContext
from pyspark import SparkConf

conf = SparkConf()
conf.set("YARN_CONF_DIR ", "D:\javaPackages\hadoop_conf_dir\yarn-conf")
conf.set("spark.driver.memory", "2g")

#conf.set("spark.executor.memory", "1g")
#conf.set("spark.python.worker.memory", "1g")
conf.setMaster("yarn-client")
conf.setAppName("TestDecisionTree")
logger = logging.getLogger('pyspark')
sc = SparkContext(conf=conf)

mylog = []
#载入和解析数据文件为 LabeledPoint RDDdata = MLUtils.loadLibSVMFile(sc,"/home/xiatao/machine_learing/")
#将数据拆分成训练集合测试集
(trainingData,testData) = data.randomSplit([0.7,0.3])

##训练决策树模型
#空的 categoricalFeauresInfo 代表了所有的特征都是连续的
model = DecisionTree.trainClassifier(trainingData, numClasses=2,categoricalFeaturesInfo={},impurity='gini',maxDepth=5,maxBins=32)

# 在测试实例上评估模型并计算测试误差

predictions = model.predict(testData.map(lambda x:x.features))
labelsAndPoint = testData.map(lambda lp:lp.label).zip(predictions)
testMSE = labelsAndPoint.map(lambda (v,p):(v-p)**2).sum()/float(testData.count())
mylog.append("测试误差是")
mylog.append(testMSE)

#存储模型

model.save(sc,"/home/xiatao/machine_learing/")
sc.parallelize(mylog).saveAsTextFile("/home/xiatao/machine_learing/log")
sameModel = DecisionTreeModel.load(sc,"/home/xiatao/machine_learing/")

```
