---
layout: post
title: spark 测试
description: spark计算框架
category: 大数据
---

## spark overview

### UC Berkeley 的spark数据分析栈
![](http://i.imgur.com/OWyTF1M.png)

按使用方式划分

- 离线批处理（Mlib，Graphs）
- 交互式查询（spark SQL）
- 时实计算（spark streaming）

### spark资源调度
![](http://i.imgur.com/rtY99ub.png)

- stanalone
-  mesos
-  yarn


  其中我们使用的是yarn资源调度，也就是运行spark job向集群申请资源的方式与hadoop是一样的，先向resourcemanger，然后在nodemanager，申请container启动applicationMaster,运行excutor

  yarn的提交job方式client和cluster
   - client提交方式，driver program运行在提交机器上
   - cluster方式，driver program是运行在集群中的某个worker中


### spark VS hadoop
- 应用场景
   
   - hadoop的mapreduce适合做大数据集的离线批处理，
   - hadoop不是万能的，小数据集（单机能处理的小数据集杀鸡用牛刀），以及复杂的迭代运算，实时计算，在线分析等无能为力，而spark的出现很好的弥补了hadoop的不足之处，因为spark是基于内存的计算框架，适合复杂的迭代计算，spark streaming弥补实时计算的空缺（storm实时性更高，吞吐量，容错方面缺不如spark，稍后介绍spark的容错机制lineage和实时计算与storm的对比）

- 运行效率
  - spark官网效率比较
  ![](http://i.imgur.com/nTBmin9.png)
  - 咱门研究中心同事实际的测试报告

```
   Spark性性能能与与MR相相比比较较提提高高了了13.6% 
    结果分析 
    之前的Hadoop版本的批处理作业，共有23个作业，作业之间的关联方式为，
 前一个作业的输出结果保存在特定目录中，作为之后作业的输入数据。其中有一定量的计算结果是仅作为中间的临时数据存在，
所有作业结束后将会被清理。这是由于每个Hadoop作业仅能执行一个MapReduce过程，
这个问 题通过Spark的编程结构可以改善为按功能模块进行作业划分，每个作业中实现多个原来MapReduce的功能，
将中间数据输出到磁盘并在下一次作业中重新读入的过程简化为Spark中的中间缓存变量保存在内存中。
这里性能的提升主要来自于此，即优化了中间数据的冗余磁盘IO时间。此外，对于省网的分析作业而言，
有着如下的特点，导致了性能提升不能达到理论上提及的一个数量级的改善效果。 
    第一，原始数据量大，输入的基础数据量过于巨大，导致大量的时间花费在第一次的磁盘数据读取上，
这个时间只取决于磁盘IO速率和文件大 小，而与分布式计算模式无关。省网分析作业内容大多属于磁盘密集型，
与数据读取的时间相比，计算的时间耗费比重较轻，使得Hadoop和Spark的性能表现差异不大。 
    第二，省网数据分析的内容，大多数属于单次的计算分析，即统计次数和汇总的工作，
这方面Hadoop的性能可以极好的发挥出来。Spark更优势 于对一组小规模输入数据的，反复迭代计算，输入文件的读取时间较小，
而计算过程十分复杂，这样其基于内存的计算方法可以更充分的展现优势。这在前一阶段中，
使用分布式对矩阵进行计算的过程中体现的尤为明显，效果可以接近理论中提及的一个数量级提升。 

```

- 开发效率比较
  - spark基于rdd的操作，是mapreduce的超集，提供我们基于rdd丰富的接口，如filter，disinct，reducebykey等等，而hadoop这些操作需要用户在map或reduce，combine自己编码实现，
  - 咱门写mapreduce程序，每个job都要写maper类，reducer类（当然有些job可以不写reducer类，如sqoop导入数据库就只需maper），可能还要写partition，combiner类，而且写完job后，需要构建job与job之间执行的顺序和依赖关系，输入输出的键值类型等；
   - 而spark是不需要这么琐碎，对rdd执行多个transform后，当执行一个action动作后（后面将介绍rdd的操作），自动构建一个基于rdd的DAG有向无环执行作业图，使用过pig的同事有所体会，这点类似pig，pig的解释器会将基于数据集的流处理过程，转换为DAG的job链，但spark又优于pig，可以做到过程控制，pig作为一个数据流语言，缺乏过程控制，粗糙的过程控制需要一门动态的脚本语言如python，javascript来实现，而且pig，hive只适合做统计分析作业，面对复杂的处理，如dougelas参数区线的压缩，需要用mapreduce或spark处理。

### 开发语言支持
 - 原生语言scala
 - java
 - python
 - spark1.4后支持R语言

### spark的核心RDD
  大家可以理解弹性分布式集合就是一个数据集合，这个集合有多个partition组成，而这些partition分布到集群中各节点的worker

 - 创建RDD的方式
  - 基于内存集合
    如1到100数字Range作为rdd，val data = sc.parallelize(1 to 100)
  - 外部存储系统，如hbase，cassandra，hdfs等， 如val data = sc.textfile("dataPath")
  
 - 基于rdd的操作
   - Transformations操作
   如map，filter，groupbykey等等，更多操作可参考[spark官网](http://spark.apache.org/docs/latest/programming-guide.html#transformations)
   
   - action操作
    top，count，reducebykey，saveastexrfile等等，更多操作可参考[spark官网](http://spark.apache.org/docs/latest/programming-guide.html#actions)
      

  transform是lazy执行的，也就是说直到遇到该rdd链执行action操作，才会启动job，执行计算，这种思想跟scala语言的lazy十分相似，下面通过一个简单的scala例子体会下这种思想
   
 
```

package com.haohandata.dnsApp
import scala.io.Source._
import scala.io.Source
/**
 * @author xizhououyang@163.com
 * @desription lazy deamon
 */
object LazyDeamon {
  /*
代码解释：当我们输入一个不存在的文件，如果不执行for循环对文件进行读取，program并不会抛异常，也就是说定义一个变量为lazy
后，当我们对其引用求值时候，才会加载运行，这点类似于java的反射机制，动态加载
*/
  def  main(args:Array[String]){
  lazy  val  file = Source.fromFile("/home/osun/pgadmin.logxx")
  for(line<- file.getLines)
    println(line)
  val word ="learning spark"
   println(word)
  
  }

}

```


### 广播变量

广播变量是分发到每个worker的只读变量不能修改，功能与hadoop的分布式缓存类似，

  目前的dns项目实战使用到是做资源表关联（大数据集与小数据集的关联），存放广播变量中，通过map转换操作做关联，注意广播变量是一个只读变量，不能做修改。

### 计数器

作业中全局的一个计数器，与hadoop的计数器类似，并不陌生，我们平时跑完mr或者pig的时候会有三种类型计数器的统计，
Framkework计数器，job计数器，hdfs文件系统计数器，注意spark中的计数器是不能在task中求值，只能在driver program中求值
   
  在dns项目中统计各用户群，各运营商，top10的icp，每个icp下统计top10 的host，可先在每个partition中统计top10的icp和top10的host，然后保存到计数器变量中，然后将聚合后结果话单过滤只保留掉计数器中的host和icp，这样可以避免多次迭代调用rdd.top（10）产生N*N个job；取五分钟小片数据，采用n*n迭代调用rdd.top方式生成库表需要两个小时，并产生了1800多个小job，跑了两个多小时，采用计数器过滤方式，4分多钟就能跑完库表实现入库postgresql

### rdd依赖
- narrow依赖（父rdd的同一个partion最多只给子rdd一个partion依赖）
- wide依赖（父rdd的同一个partion被子rdd多个partion依赖）
![](http://i.imgur.com/UE5Od8S.png)

### 小结，从计算，存储，容错谈谈rdd

- 计算

![spark计算code](http://gitlab.hudoumiao.com/TopLevel/Knowledge_Base/uploads/ba527855ed3b360d8c82840c62b0b3ab/spark计算code.png)

注意：由于时间关系，直接截了他人画的图，deamon中存在一点error，正确的代码应该是map(parts=>(parts(0),parts(1).toInt)),第一次map的transform得到的是RDD[Array[String]],不是RDD[List[String]]

code.png)![spark计算code依赖](http://gitlab.hudoumiao.com/TopLevel/Knowledge_Base/uploads/e82db4ea22a47be62bc7355505d06ba2/spark计算code依赖.png)

 
每个job划分不同的stage，每个stage就是一个Set[task]集合  

![spark提交作业流程](http://gitlab.hudoumiao.com/TopLevel/Knowledge_Base/uploads/8f0a48da38c9da65b84ed5af5262562f/spark提交作业流程.png)

  spark的作业调度，分DAGshedule，和taskshedule二级，跟hadoop的jobtraker，tasktracker两级调度类似
- 存储
 
 - MEMORY_ONLY
 - MEMORY_AND_DISK
 - MEMORY_ONLY_SER
 - MEMORY_AND_DISK_SER
 - DISK_ONLY
 - MEMORY_ONLY_2, MEMORY_AND_DISK_2

  上面是spark是rdd的各种存储策略，是spark计算框架中，默认认为重复计算rdd需要的时间会比从磁盘中读取数据进行的io操作效率高，
 因此默认所有的rdd的persist方式都是存在内存中，当内存不足后，会丢弃掉这个rdd，需要时候再根据lineage
机制从新计算，实际开发中那如果认为计算出来的rdd代价远比进行io大，这时可根据情况选择其他持久化策略，如在dns项目中，需要关联ppp的result和record话单后的rdd，采取MEMORY_AND_DISK_SER方式的持久化
- 容错（lineage）

穿插一个小故事：

```

  高帅富小明家有一个家传宝，祖训这个宝物得一代代往下传，每代人可能对这个传家宝，
每代人需对这宝物进行雕塑改造，如嵌入宝石，或者砖石，某天小明炒股亏空了，于是他要变卖这个传家宝，
可是造化弄人，当他要变卖时候，发现传家宝不见了，聪明的小明，首先会确认他爸爸是否已经把这件宝物传了给他，
如果确定是，他会将在翻遍自己房子找，如果他父亲没传给他，直接去他父亲的住处找，按照这个步骤，
如果祖父那还没找到，他会一直回溯到他曾祖父那，直到找到传家宝，然后再一代代地传给小明，
小明得到宝物后最终把它变卖

``` 

分析情景:
- rdd就好比传家宝
- 情景中的每个人物就好比不同时候集群中的计算节点中的worker
-  小明变卖宝物，就好比执行了一个action，触发提交job
-  而每代人对宝物加入一个宝石，就好比rdd的transform操作
-  rdd的容错是lineage机制，如果当向spark提交job的时候，会构造基于rdd操作的DAG的作业流，这时会有基于rdd依赖链，如果计算过程中某个rdd丢失了，它会从父rdd那重新计算，如果父rdd不存在，会一直回溯上去直到找到父的rdd，然后再依照依赖链重新执行计算，最后执行action操作


## spark在项目的实战应用
### 架构图
![](http://i.imgur.com/VdB2LPU.png)
### 项目代码

http://gitlab.hudoumiao.com/applications/User_Mobility_Analysis/tree/master/sparkcode/src/main/scala/com/haohandata


## spark streaming

### spark streaming vs Storm（下面是引用研究中心同事的给出的两者对比的报告内容）
 
```
 Storm和Spark Streaming都是分布式流处理的开源框架。虽然二者功能类似，但是也有着一定的区别。 

- 处理模型 

虽然这两个框架都提供可扩展性和容错性,它们根本的区别在于他们的处理模型。 
Spark Streaming是将流式计算分解成一系列短小的批处理作业。这里的批处理引擎是Spark，
也就是把Spark Streaming的输入数据按照batch size  （如1秒）分成一段一段的数据 （Discretized Stream），
每一段数据都 转换成Spark中的RDD  （Resilient Distributed Dataset ），然后将Spark Streaming 中对DStream的Transformation操作变为针对Spark中对RDD的Transformation操作，将RDD经过操作变成中间结果保存在内存中。整个流式计算根据业务的需求可以对中间的结果进行叠加，或者存储到外部设备。 
 
在Storm中，先要设计一个用于实时计算的图状结构，我们称之为拓扑 （topology ）。
这个拓扑将会被提交给集群，由集群中的主控节点 （masternode ）分发代码，将任务分配给工作节点 （worker node ）执行。
一个拓扑中包括spout和bolt两种角色，其中spout发送消息，负责将数据流以tuple元 组的形式发送出去；
而bolt则负责转发数据流，在bolt 中可以完成计算、过滤等操作，bolt 自身也可以随机将数据发送给其他bolt 。
在storm中，每个都是tuple是不可变数组，对应着固定的键值对。 简而言之，Storm是让数据面向计算，
而Spark Streaming是使计算面向数据。 

- 延迟，storm更高
Spark Streaming，最小的Batch Size的选取在0.5~2秒钟之间，而Storm 目前最小的延迟是100ms左右，
所以Spark Streaming能，够满足除对实时性要求非常高 （如高频实时交易）之外的所有流式准实时计算场景，
而高实时性要求的场景则应该交给Storm来完成。 

- 容错，spark streaming更好 

在容错数据保证方面的权衡是，Spark Streaming提供了更好的支持容错状态计算。
在Storm中,每个单独的记录当它通过系统时必须被跟踪，所以 Storm能够至少保证每个记录将被处理一次，
但是在从错误中恢复过来时候允许出现重复记录。这意味着可变状态可能不正确地被更新两次。 
另一方 面，Spark Streaming只需要在批级别进行跟踪处理，因此可以有效地保证每个mini-batch将完全被处理一次，
即便一个节点发生故障。 

- 吞吐量，spark streaming更强 
   Spark 目前在EC2上已能够线性扩展到100个节点 （每个节点4Core ），可以以数秒的延迟处理6GB/s的数据量 （60M records/s ），其吞吐量也比流行的Storm高2～5倍。 



使用选择 

如果你想要的是一个允许增量计算的高速事件处理系统，Storm会是最佳选择。
它可以应对你在客户端等待结果的同时，进一步进行分布式计算的需求，使用开箱即用的分布式RPC  （DRPC）就可以了。
最后但同样重要的原因：Storm使用Apache Thrift ，你可以用任何编程语言来编写拓扑结构。
如果你需要状态持续，同时/或者达到恰好一次的传递效果，应当看看更高层面的Trdent API，它同时也提供了微批处理的方式。 
如果你必须有状态的计算，恰好一次的递送，并且不介意高延迟的话，那么可以考虑Spark Streaming，
特别如果你还计划图形操作、机器学习或者访问SQL的话，ApacheSpark的stack允许你将一些library与数据流相结合 
（Spark SQL，Mllib，GraphX），它们会提供便捷的一体化编程模型。
尤其是数据流算法 （例如：K均值流媒体）允许Spark实时决策的促进。 

```

### 核心DStream
- Dstream简介
  Dstream是一组以时间为轴连续的一组rdd
![](http://i.imgur.com/H5GA2XL.png)
- Dstream的输入源

![](http://i.imgur.com/ya40qiL.png)

- DStream的transformations操作
- DSstream的action操作

### 使用场景划分
-  无状态

每次批处理，receiver接收的数据都作为数据Dstream操作

-  有状态updateStateByKey(func)

  本次计算，需要用到上次批处理的结果。
比如spark streaming的批处理时间是五分钟，但业务中，我需要统计话单中haohandata.com.cn从程序运行后，每五分钟后haohandata.com.cn这个域名的累加的访问数，这时我们会以上次批处理为key的访问次数，加上本次五分钟批处理得到结果

-  windowns

基于窗口的操作，批处理时间，滑动窗口，窗口大小
DNS实时计算实验项目中，统计五分钟粒度各rcode的次分布，
由于存在边界数据，解决的办法采取五分钟为批处理时间，滑动窗口为五分钟，窗口大小为10分钟，每次进行reduceByKeyAndWindow后，会进行过滤，只存这个windown中的中间五分钟数据，再入库cassandra

## dns项目的spark streaming实时计算（实验性项目）
### DNS项目处理流程图
![](http://i.imgur.com/7EatbDK.png)

### 项目代码



   




