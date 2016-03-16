---
layout: post
title: 大数据：kafka常见问题
description: 大数据专题
category: blog
---
    
    
## 一 kafka如何处理消费过的数据  


## 1.1 	如果想消费已经被消费过的数据    
    
   + consumer是底层采用的是一个阻塞队列，只要一有producer生产数据，那consumer就会将数据消费。当然这里会产生一个很严重的问题，如果你重启一消费者程序，那你连一条数据都抓不到，但是log文件中明明可以看到所有数据都好好的存在。换句话说，一旦你消费过这些数据，那你就无法再次用同一个groupid消费同一组数据了。    
   
   + **原因:** 消费者消费了数据并不从队列中移除，只是记录了offset偏移量。同一个consumer group的所有consumer合起来消费一个topic，并且他们每次消费的时候都会保存一个offset参数在zookeeper的root上。如果此时某个consumer挂了或者新增一个consumer进程，将会触发kafka的负载均衡，暂时性的重启所有consumer，重新分配哪个consumer去消费哪个partition，然后再继续通过保存在zookeeper上的offset参数继续读取数据。注意:offset保存的是consumer 组消费的消息偏移。    
   + 如何消费同一组数据：
     1. 采用不同的group
     2. 通过一些配置，就可以将线上产生的数据同步到镜像中去，然后再由特定的集群区处理大批量的数据。详见[详细](http://my.oschina.net/ielts0909/blog/110280)
     ![图片](/images/blog/kafka-question1.jpg)

## 1.2	如何自定义去消费已经消费过的数据
    
###  1.2.1 Conosumer.properties配置文件中有两个重要参数:    
   + **auto.commit.enable**:如果为true，则consumer的消费偏移offset会被记录到zookeeper。下次consumer启动时会从此位置继续消费。
   + **auto.offset.reset**: 该参数只接受两个常量largest和Smallest,分别表示将当前offset指到日志文件的最开始位置和最近的位置。    
    如果进一步想控制时间，则需要调用Simple Consumer，自己去设置相关参数。比较重要的参数是 kafka.api.OffsetRequest.EarliestTime()和kafka.api.OffsetRequest.LatestTime()分别表示从日志（数据）的开始位置读取和只读取最新日志。    

### 1.2.2 如何使用SimpleConsumer     
  + 首先，你必须知道读哪个topic的哪个partition 
然后，找到负责该partition的broker leader，从而找到存有该partition副本的那个broker    
  + 再者，自己去写request并fetch数据.      
  + 最终，还要注意需要识别和处理broker leader的改变.    
  
  [参考1](http://stackoverflow.com/questions/14935755/how-to-get-data-from-old-offset-point-in-kafka)    
  [参考2](https://cwiki.apache.org/confluence/display/KAFKA/Committing+and+fetching+consumer+offsets+in+Kafka)     
  [完整代码](https://cwiki.apache.org/confluence/display/KAFKA/0.8.0+SimpleConsumer+Example)        
    

#  2 kafka partition和consumer数目关系    


   1. 如果consumer比partition多，是浪费，因为kafka的设计是在一个partition上是不允许并发的，所以consumer数不要大于partition数 。
   2. 如果consumer比partition少，一个consumer会对应于多个partitions，这里主要合理分配consumer数和partition数，否则会导致partition里面的数据被取的不均匀 。最好partiton数目是consumer数目的整数倍，所以partition数目很重要，比如取24，就很容易设定consumer数目 。
   3. 如果consumer从多个partition读到数据，不保证数据间的顺序性，kafka只保证在一个partition上数据是有序的，但多个partition，根据你读的顺序会有不同 
   4. 增减consumer，broker，partition会导致rebalance，所以rebalance后consumer对应的partition会发生变化    
   
  [详见](http://www.cnblogs.com/fxjwind/p/3794255.html)     

# 3 kafka topic 副本问题    


   Kafka尽量将所有的Partition均匀分配到整个集群上。一个典型的部署方式是一个Topic的Partition数量大于Broker的数量。    
   
## 3.1 	如何分配副本
   Producer在发布消息到某个Partition时，先通过ZooKeeper找到该Partition的Leader，然后无论该Topic的Replication Factor为多少（也即该Partition有多少个Replica），Producer只将该消息发送到该Partition的Leader。Leader会将该消息写入其本地Log。每个Follower都从Leader pull数据。这种方式上，Follower存储的数据顺序与Leader保持一致.    
   
## 3.2 Kafka分配Replica的算法如下    
   1.将所有Broker（假设共n个Broker）和待分配的Partition排序.    
   2. 将第i个Partition分配到第（i mod n）个Broker上.
   3. 将第i个Partition的第j个Replica分配到第（(i + j) mode n）个Broker上.
    
   [算法详细](http://www.haokoo.com/internet/2877400.html)    

# 4 kafka如何设置生存周期与清理数据
  日志文件的删除策略非常简单:启动一个后台线程定期扫描log file列表,把保存时间超过阀值的文件直接删除(根据文件的创建时间).清理参数在server.properties文件中：
  ![](/images/blog/kafka-question2.jpg)    
  [详见](http://blog.csdn.net/lizhitao/article/details/25667831)或[官网说明](http://kafka.apache.org/documentation.html)    
  
# 5 zookeeper如何管理kafka    
  1. Producer端使用zookeeper用来"发现"broker列表,以及和Topic下每个partition leader建立socket连接并发送消息.
  2. Broker端使用zookeeper用来注册broker信息,以及监测partition leader存活性.
  3. Consumer端使用zookeeper用来注册consumer信息,其中包括consumer消费的partition列表等,同时也用来发现broker列表,并和partition leader建立socket连接,并获取消息.    
     

# 6 补充问题，kafka能否自动创建topics
  Server.properties配置文件中的一个参数:***auto.create.topics.enable=true***    
  是否自动创建    
  如果broker中没有topic的信息,当producer/consumer操作topic时,是否自动创建.  
  如果为false,则只能通过API或者command创建topic  

  
