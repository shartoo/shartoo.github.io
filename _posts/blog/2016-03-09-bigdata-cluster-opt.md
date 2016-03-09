# 0 集群优化
  一个小集群:1个master,10个datanode。    
  最开始使用pig脚本分析作业，后面作业运行时观察发现，pig脚本执行的小作业太多导致任务调度频繁，集群效率低。    
  小作业太多的影响:    
  1. 频繁新建和关闭task，频繁分配container会消耗资源。
  2. 一个oozie action先会启动一个oozie laucher作业消耗一个container，然后再启动实际的job，启动的job首先会用个container启动application master，然后在启动计算的task
现在同时最多会有29个job，至少会有50个container不是在计算。

# 1 代码优化
1. 增加5分钟基础作业时间粒度，5分钟->15分钟，减少Job数

2. 合并15分钟粒度作业，Pig->MR，grouping comparator，减少基础数据重复读取次数，减少Job数
3. 合并5分钟基础作业，一个作业处理三种话单，去除冗余字段（各粒度时间），减少Job数，减少数据量
    
# 2 集群参数配置
## 2.1 HDFS 
   + **HDFS块大小**:从默认的128MB调整为256MB，更大的块大小(block)意味着更少的job，由于当前作业计算并不复杂，可以使用更大块。
   + **复制因子**:默认是3，现在修改为2。减少数据存储量，可以减小话单上传的时间消耗
   + **DataNode处理程序计数**:参数是***dfs.datanode.handler.count*** 默认值是3，调整为30。datanode上用于处理RPC的线程数。默认为3，较大集群，可适当调大些，比如8。需要注意的是，每添加一个线程，需要的内存增加。
   + **NameNode处理程序计数**:参数是***dfs.namenode.handler.count*** 默认是30，建议值是47，现在调整为60.namenode或者jobtracker中用于处理RPC的线程数，默认是10，较大集群，可调大些，比如64。
   **NameNode服务处理程序计数**:参数是 ***dfs.namenode.service.handler.count***，默认值是30，建议值是47，现在调整为60。NameNode 用于服务调用的服务器线程数量。
   + **最大传输线程数**:参数是一起配置的为:***dfs.datanode.max.xcievers, dfs.datanode.max.transfer.threads***对于datanode来说，就如同linux上的文件句柄的限制，当datanode 上面的连接数操作配置中的设置时，datanode就会拒绝连接。
  一般都会将此参数调的很大，40000+左右。

## 2.2 YARN    
  + **每个作业的 Reduce 任务的默认数量**:参数为***mapreduce.job.reduces***默认值为1，现在调整为30。通过观察当前运行的job实例，观察其reduce执行时间，发现时间消耗不足1秒，故不必启用过多reduce。
  + **启用 Ubertask 优化**:Uber模式是Hadoop2.0针对MR小作业的优化机制。通过***mapreduce.job.ubertask.enable***来设置是否开启小作业优化，默认为false。
如果用Job足够小，则串行在的一个JVM完成该JOB，即MRAppMaster进程中，这样比为每一个任务分配Container性能更好。关于Ubertask的详细可以参考[Ubertask模式](http://qianshangding.iteye.com/blog/2259421)。
  + **Map任务内存**：参数为***mapreduce.map.memory.mb***，保持默认值1GB。
  + **Reduce任务内存**:参数为***mapreduce.reduce.memory.mb***，保持默认值1GB。
  + **Map任务CPU虚拟内核**：参数为***mapreduce.map.cpu.vcores***，为作业的每个 Map 任务分配的虚拟 CPU 内核数。默认每个map一个CPU，用户提交应用程序时，可以指定每个任务需要的虚拟CPU个数。在MRAppMaster中，每个Map Task和Reduce Task默认情况下需要的虚拟CPU个数为1。    
  + **Reduce任务CPU虚拟内核**:参数为***mapreduce.reduce.cpu.vcores***，说明 与Map任务CPU虚拟内核一致。
  + **Map 任务最大堆栈**:参数是***mapreduce.map.java.opts.max.heap***，Map 进程的最大 Java 堆栈（字节）。该参数与***mapreduce.reduce.java.opts.max.heap***一样，都是ClouderManager独有的，标准的hadoop参数是***mapreduce.map.java.opts***和***mapreduce.reduce.java.opts***
  + **Reduce 任务最大堆栈**: 同Map 任务最大堆栈。
  + **容器内存**:参数是***yarn.nodemanager.resource.memory-mb***。表示该节点上YARN可使用的物理内存总量，默认是8192（MB），注意，如果你的节点内存资源不够8GB，则需要调减小这个值，而YARN不会智能的探测节点的物理内存总量。当前配置为24GB。
  + **容器虚拟 CPU 内核**:参数是***yarn.nodemanager.resource.cpu-vcores***可以为容器分配的虚拟CPU内核的数量。集群中每台服务器只有24个虚核，所以容器内存配24G内存就行，现在作业都小map、reduce都用不了太多内存，默认是1GB。多了也没用，因为每个container至少要1个核。
 
## 2.3 Oozie
  **Oozie Server 的 Java 堆栈大小**
    默认值为1GB，现在修改为4GB。
## 2.4 HBase
  + **HBaseMaster的Java堆栈大小**:暂无调整。

  + **HBase Region Server处理程序计数**:参数为***hbase.regionserver.handler.count***,默认值为30，调节至150.是RegionServer的请求处理IO线程数。较少的IO线程，适用于处理单次请求内存消耗较高的Big PUT场景（大容量单次PUT或设置了较大cache的scan，均属于Big PUT）或ReigonServer的内存比较紧张的场景。
较多的IO线程，适用于单次请求内存消耗低，TPS要求非常高的场景。设置该值的时候，以监控内存为主要参考。
这里需要注意的是如果server的region数量很少，大量的请求都落在一个region上，因快速充满memstore触发flush导致的读写锁会影响全局TPS，不是IO线程数越高越好。
压测时，开启Enabling RPC-level logging，可以同时监控每次请求的内存消耗和GC的状况，最后通过多次压测结果来合理调节IO线程数。
  
  + **HBase RegionServer的Java堆栈大小(字节）**:HBase regionserver堆栈能多大就多大，计算方式是RegionServer java堆大小= 服务器总内存-已分配内存 （注意：此配置为优化索引入库）

## 2.5 服务器参数
  
 + 服务器时钟同步
 + 修改swappiness值
    在所有服务器上，使用root用户执行    
    ```
    # sysctl vm.swappiness=0
    # echo 'vm.swappiness=0'>> /etc/sysctl.conf
    # sysctl -p
    ```
 + 禁用透明巨页    
    ``` 
    # echo never >/sys/kernel/mm/redhat_transparent_hugepage/enabled
    ```
    关于透明巨页，参考[透明巨页](http://blog.chinaunix.net/uid-26489617-id-3205109.html)
