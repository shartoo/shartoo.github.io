---
layout: post
title: 使用pyspark做数据挖掘
description: 大数据
category: 大数据
---

## 一 环境准备

### 1.1 编程环境

    必须加入spark内容，将以下代码加入推荐逻辑之前   

```
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
```

各个参数视个人机器配置而定

### 1.2  本地模式

    在本地模式运行时，参数需要如下设定

```
conf.setMaster("client")
此时代码会在本地执行，对于小数据集（小于1MB的数据）可以正常执行，如果数据超过1MB将会遇到以下问题：
ERROR PythonRDD: Python worker exited unexpectedly (crashed)
java.net.SocketException: Connection reset by peer: socket write error
        at java.net.SocketOutputStream.socketWrite0(Native Method)
解决办法参考“pyspark处理大数据集” 这篇笔记
```

### 1. 3 数据准备

    数据都是存放在HDFS上，需要先将数据上传到个人目录。

## 二 数据挖掘视角

    接下来进入数据挖掘的思路，数据挖掘标准流程如下：
    
+ 数据收集（本案例中数据已经准备好）

+ 数据清洗转换

+ 根据数据选择算法模型（本案例以推荐算法为例）

+ 训练模型：使用训练数据训练算法模型中一些参数。

+ 使用模型：使用训练好的模型预测或者对检验数据分类、聚类

+ 评估模型：验证模型预测结果与真实结果误差，评估准确率、召回率等指标

### 2.1 收集数据

```
dataset_path = os.path.join('/home/xiatao/machine_learing/moive_recommend/','')
complete_dataset_path =   os.path.join(dataset_path,'ml-latest.zip')
small_dataset_path = os.path.join(dataset_path,'ml-latest-small.zip')
```

### 2.2 数据清洗转换

   本示例中数据清洗和转换比较简单，只是去掉数据头，并将数据封装成RDD。在其他数据中可能需要去除部分没用的列，数据降维，连续型数据转换为离散型等操作。
```   
# 载入数据，将数据的头过滤出来
small_rating_file = os.path.join(dataset_path,'latest_small','ratings.csv')
small_rating_raw_data = sc.textFile(small_rating_file)
small_rating_raw_data_header = small_rating_raw_data.take(1)[0]
# 将原始数据封装成新的RDD
small_rating_data  = small_rating_raw_data.filter(lambda line:line!=small_rating_raw_data_header)\
    .map(lambda line:line.split(",")).map(lambda tokens:(tokens[0],tokens[1],tokens[2])).cache()
```

### 2.3 根据数据选择算法模型

    本示例以推荐算法中的ALS（最小交替二乘法）为例，关于交替二乘法参考 https://www.zhihu.com/question/31509438
    
### 2.4  训练模型

    在开始之前，我们先将数据分为三份，分别是training_rdd（训练数据） ,validation_rdd（验证数据）,test_rdd（测试数据）。使用training_rdd获得一个训练模型，然后去预测validation_rdd结果，并计算训练模型对validation_rdd预测结果与validation_rdd真实结果之间误差，以此来决定模型应该使用的参数。

```
# 使用小数据集选择 交叉最小二乘法参数
# 首先将数据分为训练数据，校验数据，测试数据
training_rdd ,validation_rdd,test_rdd = small_rating_data.randomSplit([6,2,2],seed=0L)
validation_for_predict_rdd = validation_rdd.map(lambda x:(x[0],x[1]))
test_for_predict_rdd =test_rdd.map(lambda x:(x[0],x[1]))
```

 推荐算法ALS中最重要的比较重要的参数有如下：
    
+ rank：特征向量秩大小，越大的秩会得到更好的模型，但是计算消耗也相应增加。默认是 10

+ iteration： 算法迭代次数（默认是10）

+ lambda：正则参数，默认是 0.01。详细解释参考

+ alpha：在隐式ALS中用于计算置信度的常量，默认为1.0

+ numUserBlocks,numProductBlocks：将用户和产品数据分解的块数目，用来控制并行度；你可以传入-1来让MLlib自动决定。
    
     本示例中，算法迭代次数固定为10（可以根据实际情况调整），lambda参数固定位 0.1 ，由于本文的电影评分数据是确信数据，使用的ALS是确定模型，因此不需要alpha参数，numUserBlocks和numProductBlocks参数使用默认参数。
    因此，本示例中需要训练的参数是 rank，不同的rank会影响模型的预测准确度，不同的模型其预测误差可以通过均方根误差（标准方差RMSE）来衡量优劣。选取均方根误差最小的rank作为预测模型的rank。

```
seed =5L
iterations =10
regularization_parmeter =0.1
ranks =[4,5,6,7,8,10,12]
errors = [0,0,0,0,0,0,0]
err =0
min_error = float('inf')
best_rank =-1
best_interation =-1
for rank in ranks:
    model = ALS.train(training_rdd,rank,seed=seed,iterations=iterations,lambda_=regularization_parmeter)
    predictions = model.predictAll(validation_for_predict_rdd).map(lambda r:((r[0],r[1],r[2])))
    rates_and_preds =validation_rdd.map(lambda r:(int(r[0]),int(r[1]),float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r:(r[1][0]-r[1][1])**2).mean())
    errors[err] = error
    err+=1
    print 'For rank %s the RMSE is %s'%(rank,error)
    if error < min_error:
        min_error =error
        best_rank = rank
```

## 2.5 使用模型

    使用训练数据中获得的最佳参数来构建新的推荐模型，本示例中使用完整数据集ml-lates数据集检验模型预测结果,为了检验模型准确率，我们将数据分为训练数据和验证数据两份，分别为training_complete_rdd(70%),test_complete_rdd (30%)

```
#现在开始使用完整数据集来构建最终模型
complete_rating_file = os.path.join(dataset_path,'latest_all','ratings.csv')
complete_rating_raw_data =sc.textFile(complete_rating_file)
complete_rating_raw_data_header = complete_rating_raw_data.take(1)[0]
complete_rating_data = complete_rating_raw_data.filter(lambda line:line!=complete_rating_raw_data_header)\
    .map(lambda line:line.split(",")).map(lambda tokens:(int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()
#现在开始训练推荐模型
training_complete_rdd,test_complete_rdd = complete_rating_data.randomSplit([7,3],seed =0L)
complete_model = ALS.train(training_complete_rdd,best_rank,seed = seed,iterations=\
    iterations,lambda_ =regularization_parmeter)
```

### 2.6 评估验证模型


     使用完整数据集中30%的部分来测试模型预测结果准确率。 
```
#在测试数据集上测试
test_for_predict_rdd = test_complete_rdd.map(lambda x:(x[0],x[1]))
predictions_complete =complete_model.predictAll(test_for_predict_rdd).map(lambda r:((r[0],r[1],r[2])))
rates_and_preds_complete = test_complete_rdd.map(lambda r:((int(r[0]),int(r[1])),float(r[2]))).join(predictions_complete)
error_complete = math.sqrt(rates_and_preds_complete.map(lambda r: (r[1][0]-r[1][1]) **2).mean())
mylog.append( "完整数据集的误差是RMSE   %s"%(error_complete))
```

    此示例中只使用了平方根误差来评估模型。

### 2.7 模型后续使用

#### 2.7.1 给老用户（对部分电影有评分）推荐

    添加新数据，每次添加新数据都需要重新训练模型，此时将新数据与原数据合并再训练并得到模型。
```
#添加新的用户评分，
new_user_ID = 0
new_user_rating =[
    (0,260,9),
    (0,1,8),
    (0,16,7),
    (0,25,8),
    (0,32,9),
    (0,335,4),
    (0,379,4),
    (0,296,4),
    (0,854,10),
    (0,50,8)
]
new_user_rating_RDD = sc.parallelize(new_user_rating)
mylog.append( "新用户的评分是 %s"%new_user_rating_RDD.take(10))
# 将数据加入到推荐模型将使用的训练数据中，
complete_data_with_new_rating_RDD = complete_rating_data.union(new_user_rating_RDD)
# 最后，使用前面选择的最优参数来训练ALS模型
from time import time
new_rating_model = ALS.train(complete_data_with_new_rating_RDD,best_rank,seed = seed,iterations=iterations,lambda_ =regularization_parmeter)
```

再利用此模型向老用户推荐电影   

```
# 获取最好的推荐。鉴于我们将获得新用户没有评分的RDD
#获得电影ID
new_users_ratings_ids = map(lambda x:x[1],new_user_rating)
#获得不在ID列表中的
new_user_unrated_moive_RDD = (complete_moive_data.filter(lambda x:x[0] not in new_users_ratings_ids)\
                              .map(lambda x:(new_user_ID,x[0])))
# 使用输入的RDD和 new_user_unrated_moive_RDD，使用 new_rating_mode.predictAll() 来预测电影
new_user_recommendations_RDD = new_rating_model.predictAll(new_user_unrated_moive_RDD)
new_user_recommendations_rating_RDD = new_user_recommendations_RDD.map(lambda x:(x.product,x.rating))
new_user_recommendations_rating_title_and_count_RDD =new_user_recommendations_rating_RDD.join(complete_moive_titles)\
    .join(moive_rating_counts_RDD)
new_user_recommendations_rating_title_and_count_RDD.take(3)
top_moives = new_user_recommendations_rating_title_and_count_RDD.map(lambda r:(r[1][0][1],r[1][0][0],r[1][1]))\
    .filter(lambda r:r[2]>=25).takeOrdered(25,key=lambda x:-x[1])
mylog.append( "推荐的电影（浏览量超过25的）%s"%'\n'.join(map(str,top_moives)))
```

#### 2.7.2  预测新用户对某部电影评分

## 三  模型的保存于复用

     可以将我们的模型存储起来作为后续的在线推荐系统使用，尽管每次有新的用户评分数据时都会生成新的模型，为了节省服务启动时间。当前的模型也是值得存储的。我们可以通过存储那些RDD以节省时间，尤其是那些需要消耗极大时间的。

```   
from pyspark.mllib.recommendation import MatrixFactorizationModel
model_path = os.path.join(dataset_path,'models','moive_lens_als')
model.save(sc,model_path)
same_model = MatrixFactorizationModel.load(sc,model_path)
```




 
