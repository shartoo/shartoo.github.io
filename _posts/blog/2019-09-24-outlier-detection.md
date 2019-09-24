---
layout: post
title: 使用pyod做离群点检测
description: 数据科学
category: blog
---

https://www.analyticsvidhya.com/blog/2019/02/outlier-detection-python-pyod/

### 1 什么是离群点

一个极大的偏离正常值的数据点。下面是一些常见的离群点

+ 一个学生的平均得分超过剩下90%的得分，而其他人的得分平均仅为70%。显然的离群点
+ 分析某个顾客的购买行为，大部分集中在100-块，突然出现1000块的消费。

有多种类型的离群点
+ **单变量的**： 只有一个变量的值会出现极端值
+ **多变量的**： 至少两个以上的变量值的综合得分极端。


### 2 为什么需要检测离群点

离群点会影响我们的正常的数据分析和建模，如下图左边是包含离群点的模型，右边是处理掉离群点之后的模型结构。

![](/images/blog/outlier_sample.png)

但是，**离群点并非一直都是不好的**。简单的移除离群点并非明智之举，我们需要去理解离群点。

现在的趋势是使用直接的方式如盒图、直方图和散点图来检测离群点。但是**在处理大规模数据集和需要在更大数据集中识别某种模式时，专用的离群点检测算法是非常有价值的**。

某些应用，如金融欺诈识别和网络安全里面的入侵检测需要及时响应的以及精确的技术来识别离群点。

### 3 为什么要使用PyOD 来做离群点检测

现有的一些实现，比如PyNomaly，并非为了做离群点而设计的（尽管依然值得一试）。PyOD是一个可拓展的Python工具包用来检测多变量数据中的离群点。提供了接近20种离群点检测算法。

### 4 PyOD的特征

+ 开源、并附有详细的说明文档和实例。
+ 支持先进的模型，包括神经网络，深度学习和离群点检测集成学习方法
+ 使用JIT优化加速，以及使用numba和joblib并行化
+ python2 和3 都可以用

### 5 安装使用PyOD

```
pip install pyod
pip install --upgrade pyod # to make sure that the latest version is installed!
```
注意，PyOD包含了一些神经网络模型，基于keras。但是它不会自动安装Keras或者Tensorflow。需要手动安装这两个库，才能使用其神经网络模型。安装过程的依赖有点多。

### 6 使用PyOD来做离群点检测

注意，我们使用的是**离群得分**，即每个模型都会给每个数据点打分而非直接根据一个阈值判定某个点是否为离群点。

**Angle_Based Outlier Detection (ABOD)**

  - 它考虑了每个数据点和其邻居的关系，但是不考虑邻居之间的关系。
+ ABOD在多维度数据上表现较好
+ PyOD提供了两种不同版本的ABOD
  - Fast ABOD：使用KNN来近似
  - Original ABOD：以高时间复杂度来考虑所有训练数据点

**KNN 检测器**

+ 对于任意数据点，其到第k个邻居的距离可以作为其离群得分
+ PyOD提供三种不同的KNN检测器
  -  `Largest`： 使用第k个邻居的距离来作为离群得分
  -  `Mean`: 使用全部k个邻居的平均距离作为离群得分
  -  `Median`:使用k个邻居的距离的中位数作为离群得分

**Isolation Forest**

+ 内部使用sklearn，此方法中，使用一个集合的树来完成数据分区。孤立森林提供农一个离群得分来判定一个数据点在结构中有多孤立。其离群得分用来将它与正常观测数据区分开来。
+ 孤立森林在多维数据上表现很好

**Histogram-based Outiler Detection**

+ 一种高效的无监督方法，它假设特征之间独立，然后通过构建直方图来计算离群得分
+ 比多变量方法快得多，但是要损失一些精度

**Local Correlation Integral(LOCI)**

+ LOCI在离群检测和离群点分组上十分高效。它为每个数据点提供一个LOCI plot，该plot反映了数据点在附近数据点的诸多信息，确定集群、微集群、它们的半径以及它们的内部集群距离
+ 现存的所有离群检测算法都无法超越此特性，因为他们的输出仅仅是给每个数据点的输出一个单一值。

**Feature Bagging**

+ 一个特征集合检测器，它在数据集的一系列子集上拟合了大量的基准检测器。它使用平均或者其他结合方法来提高预测准确率
+ 默认使用LOF(Local Outiler Factor)作为基准评估器。但是其他检测器，如KNN，ABOD都可以作为基准检测器
+ Feature Bagging首先通过随机选取特征子集来构建n个子样本。这带来了基准评估器的多样性。最终，通过取所有基准评估器的平均或者最大值来预测得分。

***Clustering Based  Local Outiler Factor**

+ 它将数据分为小聚类簇和大聚类簇。离群得分基于数据点所属的聚类簇的大小来计算，距离计算方式为到最近大聚类簇的距离。

### 7 PyOD在 Big Mart Sales 问题上的表现

[Big Mart Sales Problem](https://datahack.analyticsvidhya.com/contest/practice-problem-big-mart-sales-iii/?utm_source=outlierdetectionpyod&utm_medium=blog)。需要注册然后下载数据集，附件中有

```
import pandas as pd
import numpy as np
from scipy import stats
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm

df = pd.read_csv("train.csv")
print(df.describe())
show = False
if show:
    plt.figure(figsize=(10,10))
    plt.scatter(df['Item_MRP'],df['Item_Outlet_Sales'])
    plt.xlabel("Item_MRF")
    plt.ylabel("Item_Outlet_Sales")
    plt.show()

scaler = MinMaxScaler(feature_range=(0,1))
df[['Item_MRP','Item_Outlet_Sales']] = scaler.fit_transform(df[['Item_MRP','Item_Outlet_Sales']])
print(df[['Item_MRP','Item_Outlet_Sales']].head())

x1 = df['Item_MRP'].values.reshape(-1,1)
x2 = df['Item_Outlet_Sales'].values.reshape(-1,1)
x = np.concatenate((x1,x2),axis=1)
# 设置 5%的离群点数据
random_state = np.random.RandomState(42)
outliers_fraction = 0.05
# 定义7个后续会使用的离群点检测模型
classifiers = {
    "Angle-based Outlier Detector(ABOD)" : ABOD(contamination=outliers_fraction),
    "Cluster-based Local Outiler Factor (CBLOF)": CBLOF(contamination = outliers_fraction,check_estimator=False,random_state = random_state),
    "Feature Bagging" : FeatureBagging(LOF(n_neighbors=35),contamination=outliers_fraction,check_estimator=False,random_state = random_state),
    "Histogram-base Outlier Detection(HBOS)" : HBOS(contamination=outliers_fraction),
    "Isolation Forest" :IForest(contamination=outliers_fraction,random_state = random_state),
    "KNN" : KNN(contamination=outliers_fraction),
    "Average KNN" :KNN(method='mean',contamination=outliers_fraction)
}

#逐一 比较模型
xx,yy = np.meshgrid(np.linspace(0,1,200),np.linspace(0,1,200))
for i ,(clf_name,clf) in enumerate(classifiers.items()):
    clf.fit(x)
    # 预测利群得分
    scores_pred = clf.decision_function(x)*-1
    # 预测数据点是否为 离群点
    y_pred = clf.predict(x)
    n_inliers = len(y_pred)-np.count_nonzero(y_pred)
    n_outliers = np.count_nonzero(y_pred==1)
    plt.figure(figsize=(10,10))

    # 复制一份数据
    dfx = df
    dfx['outlier'] = y_pred.tolist()
    # IX1 非离群点的特征1，IX2 非利群点的特征2
    IX1 = np.array(dfx['Item_MRP'][dfx['outlier']==0]).reshape(-1,1)
    IX2 = np.array(dfx['Item_Outlet_Sales'][dfx['outlier']==0]).reshape(-1,1)
    # OX1 离群点的特征1，OX2离群点特征2
    OX1 = np.array(dfx['Item_MRP'][dfx['outlier']==1]).reshape(-1,1)
    OX2 = np.array(dfx['Item_Outlet_Sales'][dfx['outlier'] == 1]).reshape(-1, 1)
    print("模型 %s 检测到的"%clf_name,"离群点有 ",n_outliers,"非离群点有",n_inliers)

    # 判定数据点是否为离群点的 阈值
    threshold = stats.scoreatpercentile(scores_pred,100*outliers_fraction)
    # 决策函数来计算原始的每个数据点的离群点得分
    z = clf.decision_function(np.c_[xx.ravel(),yy.ravel()]) * -1
    z = z.reshape(xx.shape)
    # 最小离群得分和阈值之间的点 使用蓝色填充
    plt.contourf(xx,yy,z,levels=np.linspace(z.min(),threshold,7),cmap=plt.cm.Blues_r)
    # 离群得分等于阈值的数据点 使用红色填充
    a = plt.contour(xx,yy,z,levels=[threshold],linewidths =2,colors='red')
    # 离群得分在阈值和最大离群得分之间的数据 使用橘色填充
    plt.contourf(xx,yy,z,levels=[threshold,z.max()],colors='orange')
    b = plt.scatter(IX1,IX2,c='white',s=20,edgecolor ='k')
    c = plt.scatter(OX1,OX2,c='black',s=20,edgecolor = 'k')
    plt.axis('tight')
    # loc = 2 用来左上角
    plt.legend(
        [a.collections[0],b,c],
        ['learned decision function','inliers','outliers'],
        prop=mfm.FontProperties(size=20),
        loc=2
    )
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.title(clf_name)
    plt.savefig("%s.png"%clf_name)
    #plt.show()

```
结果如下；

![](/images/blog/outlier_detection_ABOD.png)
![](/images/blog/outlier_detection_avg_knn.png)
![](/images/blog/outlier_detection_CBLOF.png)
![](/images/blog/outlier_detection_feature_bagging.png)
![](/images/blog/outlier_detection_HBOS.png)
![](/images/blog/outlier_detection_Isolation_Forest.png)
![](/images/blog/outlier_detection_KNN.png)
