---
layout: post
title: elastic search 性能测试
description: 搜索&NLP
category: blog
---

# ElasticSearch性能测试 
原文翻译自:[ElasticSearch官方性能测试](https://benchmarks.elastic.co/index.html)
<hr>

# 基准测试场景
注：ES中的文档类似一条记录。
**数据**
测试使用了860万份文档，取自Geonames的POI数据。

|||
|-|-|
|文档数|860万|
|数据大小|2.8GB(JSON)|
|客户端线程数|8|
|每个bulk请求|5000份文档|
|服务器数目|1个或2个|

**服务器配置**

|||
|-|-|
|核心数|36个real cores,使用超线程可达72个|
|RAM|256|
|SSD|Intel 750 PCIe/NVMe|

**默认设置**
+ 默认，2个节点都是追加，使用全部默认配置，2个节点运行在一个沙盒中（5个shards,1份拷贝）
+ 4GB heap(ES_HEAP_SIZE)
+ refresh_interval: 30s
+ index.number_of_shards: 6
+ index.number_of_replicas:0
+ index.translog.flush_threshold_size: 4g


##索引性能
以下图片展示的是ElasticSearch每晚基于master分支代码的性能测试结果。
 ![索引性能测试](/images/blog/索引性能测试.png)
<hr>
## 时间消耗
下图显示的是与索引性能使用相同数据时，索引时间（分钟），合并时间（分钟），刷新时间（分钟），Flush时间（分钟），Merge throttle 时间（分钟）。
![时间消耗](/images/blog/时间消耗.png)
<hr>
## 合并时间，部分
![合并时间，部分](/images/blog/合并时间部分.png)
<hr>
## 堆使用的总段数
![栈使用](/images/blog/栈使用.png)
<hr>
## Indexing CPU utilization（默认）
![索引CPU](/images/blog/索引CPU.png)
<hr>
## 索引磁盘使用
![索引磁盘使用](/images/blog/索引磁盘使用.png)
<hr>
## 索引段数
![索引段数](/images/blog/索引段数.png)
<hr>
## gradle测试时间
gradle是一个基于Apache Ant和Apache Maven概念的项目自动化建构工具。它使用一种基于Groovy的特定领域语言(DSL)来声明项目设置。[gradle](http://baike.baidu.com/link?url=DTDzrjTMJX-D4eNPwemJ0RSqD1kmDvKmBVQc3x5WSYs0qEPTBni-rwmTugMtyAG1ukQlXC_m3BC0DkIbA0Uf6q)
![gradle测试时间](/images/blog/gradle测试时间.png)
<hr>
<hr>
## Search QPS(脚本)
![Search QPS 脚本](/images/blog/SearchQPS.png)
<hr>
## Search QPS
![search qps](/images/blog/SearchQPS1.png)
<hr>
## 请求时间统计
![请求时间统计](/images/blog/请求时间统计.png)
<hr>

## 垃圾回收时间
![垃圾回收时间](/images/blog/垃圾回收时间.png)

<hr>

## 
