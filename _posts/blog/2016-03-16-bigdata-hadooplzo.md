---
layout: post
title: hadoop lzo问题
description: 大数据专题
category: blog
---
    
    
## 一 重要问题   
## 1.1  hadoop-gpl-compression还是hadoop-lzo
  **hadoop-lzo-xxx** 的前身是**hadoop-gpl-compression-xxx**,之前是放在googlecode下管理,[地址](http://code.google.com/p/hadoop-gpl-compression/)但由于协议问题后来移植到github上,也就是现在的hadoop-lzo-xxx,github,[链接地址](https://github.com/kevinweil/hadoop-lzo).    
    网上介绍hadoop lzo压缩都是基于hadoop-gpl-compression的介绍.而hadoop-gpl-compression还是09年开发的,跟现在hadoop版本已经无法再完全兼容,会发生一些问题.而按照网上的方法,为了兼容hadoop,使用hadoop-lzo-xxx。

  **原理：**因为hadoop lzo实际上得依赖C/C++开发的lzo去压缩,而他们通过JNI去调用.如果使用hadoop-gpl-compression下的Native,但使用hadoop-lzo-xxx的话,会导致版本不一致问题.所以正确的做法是,将hadoop-lzo-xxx下的Native放入到/usr/local/lib下.而你每升级一个hadoop-lzo-xxx版本,或许就得重复将新lzo版本下的native目录放入/usr/local/lib下.具体需要测试.
同时这里说下,hadoop-lzo-xxx的验证原理,让我们更系统的了解为什么使用hadoop-lzo会报的一系列错误.   
1. 首先Hadoop-lzo会通过JNI调用gplcompression,如果调取不到会报Could not load native gpl library异常.具体代码如下:    

```
static {   
   try {  
       //try to load the lib      
         System.loadLibrary("gplcompression"); 
         nativeLibraryLoaded = true;  
         LOG.info("Loaded native gpl library");  
      } catch (Throwable t) {  
	  LOG.error("Could not load native gpl library", t);  
	  nativeLibraryLoaded = false;  
	 }  
```      
2. 获取了gplcompression后需要初始化加载以便可以调用,如果加载不成功,如我刚才说的版本冲突等也会报一系列错误.同时这里的加载和初始化分成两步,一步是压缩,对应Java的类是LzoCompressor.另一步解压缩,对应Java的类是LzoDecompressor.先看下LzoCompressor是如何加载初始化的,代码如下:          

```   
	static {  
	  if (GPLNativeCodeLoader.isNativeCodeLoaded()) {  
	    // Initialize the native library  
	    try {  
	      initIDs();  
	      nativeLzoLoaded = true;  
	    } catch (Throwable t) {  
	      // Ignore failure to load/initialize native-lzo  
	      LOG.warn(t.toString());  
	      nativeLzoLoaded = false;  
	    }  
	    LZO_LIBRARY_VERSION = (nativeLzoLoaded) ? 0xFFFF & getLzoLibraryVersion()  
	        : -1;  
	  } else {  
	    LOG.error("Cannot load " + LzoCompressor.class.getName() +   
	    " without native-hadoop library!");  
	    nativeLzoLoaded = false;  
	    LZO_LIBRARY_VERSION = -1;  
	  }  
	}
```      

   如我这里所报的警告    
	'WARN lzo.LzoCompressor: java.lang.NoSuchFieldError: workingMemoryBuf'    
  就是由这里的 **LOG.warn(t.toString())**所抛出.同时这里也会先加载gplcompression,加载不成功同样会报    
	without native-hadoop library!
  错误.再看看解压缩LzoDecompressor,原理差不多,不再阐述,代码如下:    
  
```
   static {  
	  if (GPLNativeCodeLoader.isNativeCodeLoaded()) {  
	    // Initialize the native library  
	    try {  
	      initIDs();  
	      nativeLzoLoaded = true;  
	    } catch (Throwable t) {  
	      // Ignore failure to load/initialize native-lzo  
	      LOG.warn(t.toString());  
	      nativeLzoLoaded = false;  
	    }  
	    LZO_LIBRARY_VERSION = (nativeLzoLoaded) ? 0xFFFF & getLzoLibraryVersion()  
	        : -1;  
	  } else {  
	    LOG.error("Cannot load " + LzoDecompressor.class.getName() +   
	    " without native-hadoop library!");  
	    nativeLzoLoaded = false;  
	    LZO_LIBRARY_VERSION = -1;  
	  }  
	}
```      
	
# 二 如何安装LZO    

1.首先下载https://github.com/kevinweil/hadoop-lzo/,我这里下载到/home/guoyun/Downloads//home/guoyun/hadoop/kevinweil-hadoop-lzo-2dd49ec
2. 去lzo源码根目录下执行    
``` 
	wget https://download.github.com/kevinweil-hadoop-lzo-2ad6654.tar.gz  
	tar -zxvf kevinweil-hadoop-lzo-2ad6654.tar.gz  
	cd kevinweil-hadoop-lzo-2ad6654 
	export CFLAGS=-m64
	export CXXFLAGS=-m64
	ant compile-native tar    
```  
2. 通过ant生成native和jar,命令如下:
  在build目录下生成对应的tar包,解压缩后,进入该目录可以看到对应的jar包hadoop-lzo-0.4.14.jar.同时将lib/native/Linux-amd64-64/目录下所有文件拷贝到$HADOOP_HOME/lib和/usr/local/lib两个目录下.
  **注明:**拷贝到/usr/local/lib是便于调试,如是生产环境则无需拷贝.    

  **注意：**如果 Hadoop/lib/目录下没有native/Linux-amd64-64/ 目录，需要手工创建。或者下载hadoop-gpl-compression。参考(http://guoyunsky.iteye.com/blog/1237327),安装步骤中的第四步，复制库文件到hadoop/lib目录下的操作。
  ```mv hadoop-gpl-compression-0.1.0/lib/native/Linux-amd64-64/* $HADOOP_HOME/lib/native/Linux-amd64-64/```
 
   
# 三 如何确定是否已经安装好LZO
  [参考](https://code.google.com/a/apache-extras.org/p/hadoop-gpl-compression/wiki/FAQ?redir=1)
 执行命令:        
 
```
 % ls -l /usr/lib*/liblzo2*
-rw-r--r--  1 root root 171056 Mar 20  2006 /usr/lib/liblzo2.a
lrwxrwxrwx  1 root root     16 Feb 17  2007 /usr/lib/liblzo2.so -> liblzo2.so.2.0.0*
lrwxrwxrwx  1 root root     16 Feb 17  2007 /usr/lib/liblzo2.so.2 -> liblzo2.so.2.0.0*
-rwxr-xr-x  1 root root 129067 Mar 20  2006 /usr/lib/liblzo2.so.2.0.0*
-rw-r--r--  1 root root 208494 Mar 20  2006 /usr/lib64/liblzo2.a
lrwxrwxrwx  1 root root     16 Feb 17  2007 /usr/lib64/liblzo2.so -> liblzo2.so.2.0.0*
lrwxrwxrwx  1 root root     16 Feb 17  2007 /usr/lib64/liblzo2.so.2 -> liblzo2.so.2.0.0*
-rwxr-xr-x  1 root root 126572 Mar 20  2006 /usr/lib64/liblzo2.so.2.0.0*    
```

 lzo压缩已经广泛用于Hadoop中,至于为什么要在Hadoop中使用Lzo.这里不再重述.其中很重要的一点就是由于分布式计算,所以需要支持对压缩数据进行分片,也就是Hadoop的InputSplit,这样才能分配给多台机器并行处理.所以这里花了一天的时间,看了下Hadoop lzo的源码,了解下Hadooplzo是如何做到的.    

 其实一直有种误解,就是以为lzo本身是支持分布式的,也就是支持压缩后的数据可以分片.我们提供给它分片的逻辑,由lzo本身控制.但看了Hadoop lzo源码才发现,lzo只是个压缩和解压缩的工具,如何分片,是由Hadooplzo(Javad代码里)控制.具体的分片算法写得也很简单,就是在内存中开一块大大的缓存,默认是256K,缓存可以在通过io.compression.codec.lzo.buffersize参数指定.数据读入缓存(实际上要更小一些),如果缓存满了,则交给lzo压缩,获取压缩后的数据,同时在lzo文件中写入压缩前后的大小以及压缩后的数据.所以这里,一个分片,其实就是<=缓存大小.具体lzo文件格式(这里针对Lzop):    
 1.lzo文件头
 + 写入lzo文件标识： 此时长度9
 + 写入版本    
    ```
	LZOP_VERSION		lzo版本，short，此时长度11
	LZO_VERSION_LIBRARY	lzo压缩库版本，short，此时长度13
	LZOP_COMPAT_VERSION	最后lzo应该一直的版本，short，此时长度15     
   ```    
  + 写入压缩策略    
  + LZO1X_1的话writeByte写入1和5，此时长度17 
  + writeInt写入flag(标识)，此时长度21    
  + writeInt写入mode(模式)，此时长度25    
  + writeInt写入当前时间秒，此时长度29    
  + writeInt写入0,不知道做何用途，此时长度33   
  + writeBye写入0，不知道做何用途，此时长度34    
  + writeInt写入之前数据的checksum，此时长度38
    

 2. 写入多个块,会有多个.循环处理,直到压缩完成
   写入压缩前的数据长度,此时长度为39如果压缩前的长度小于压缩后的长度,则写入未压缩的数据长度,再写入未压缩的数据.反之则写入压缩后的数据长度,以及压缩后的数据    
 3. lzo文件尾,只是写入4个0,不知道做什么用途    		   同时如果你指定索引文件路径的话,则一个缓存写完后便会将写入的数据长度写到索引文件中.如此在Hadoop分布式时只要根据索引文件的各个长度,读取该长度的数据 ,便可交给map处理.     
    以上是hadoop lzo大概原理,同时LzopCodec支持在压缩时又生成对应的索引文件.而LzoCodec不支持.具体代码看下来,还不明确LzoCodec为何没法做到,也是一样的切片逻辑.具体待测试.
       
# 4 hadoop中使用lzo的压缩    
 在hadoop中使用lzo的压缩算法可以减小数据的大小和数据的磁盘读写时间，不仅如此，lzo是基于block分块的，这样他就允许数据被分解成chunk，并行的被hadoop处理。这样的特点，就可以让lzo在hadoop上成为一种非常好用的压缩格式。    

   lzo本身不是splitable的，所以当数据为text格式时，用lzo压缩出来的数据当做job的输入是一个文件作为一个map。但是sequencefile本身是分块的，所以sequencefile格式的文件，再配上lzo的压缩格式，就可实现lzo文件方式的splitable。    

   由于压缩的数据通常只有原始数据的1/4，在HDFS中存储压缩数据，可以使集群能保存更多的数据，延长集群的使用寿命。不仅如此，由于mapreduce作业通常瓶颈都在IO上，存储压缩数据就意味这更少的IO操作，job运行更加的高效。但是，在hadoop上使用压缩也有两个比较麻烦的地方：   

   + 第一，有些压缩格式不能被分块，并行的处理，比如gzip。    
   + 第二，另外的一些压缩格式虽然支持分块处理，但是解压的过程非常的缓慢，使job的瓶颈转移到了cpu上，例如bzip2。比如我们有一个1.1GB的gzip文件，该文件 被分成128MB/chunk存储在hdfs上，那么它就会被分成9块。为了能够在mapreduce中并行的处理各个chunk，那么各个mapper之间就有了依赖。而第二个mapper就会在文件的某个随机的byte出进行处理。那么gzip解压时要用到的上下文字典就会为空，这就意味这gzip的压缩文件无法在hadoop上进行正确的并行处理。也就因此在hadoop上大的gzip压缩文件只能被一个mapper来单个的处理，这样就很不高效，跟不用mapreduce没有什么区别了。而另一种bzip2压缩格式，虽然bzip2的压缩非常的快，并且甚至可以被分块，但是其解压过程非常非常的缓慢，并且不能被用streaming来读取，这样也无法在hadoop中高效的使用这种压缩。即使使用，由于其解压的低效，也会使得job的瓶颈转移到cpu上去。    
  
  如果能够拥有一种压缩算法，即能够被分块，并行的处理，速度也非常的快，那就非常的理想。这种方式就是lzo。lzo的压缩文件是由许多的小的blocks组成（约256K），使的hadoop的job可以根据block的划分来splitjob。不仅如此，lzo在设计时就考虑到了效率问题，它的解压速度是gzip的两倍，这就让它能够节省很多的磁盘读写，它的压缩比的不如gzip，大约压缩出来的文件比gzip压缩的大一半，但是这样仍然比没有经过压缩的文件要节省20%-50%的存储空间，这样就可以在效率上大大的提高job执行的速度。以下是一组压缩对比数据，使用一个8.0GB的未经过压缩的数据来进行对比：    
    

|压缩格式|文件大小(GB)|压缩时间|解压时间|
|:-------:|:-----------:|:-------:|:-------:|
|None|	some_logs|	8.0|	-|	-|
|Gzip	|some_logs.gz	|1.3	|241	|72|
|LZO|	some_logs.lzo|	2.0|	55|	35|    

可以看出，lzo压缩文件会比gzip压缩文件稍微大一些，但是仍然比原始文件要小很多倍，并且lzo文件压缩的速度几乎相当于gzip的5倍，而解压的速度相当于gzip的两倍。lzo文件可以根据blockboundaries来进行分块，比如一个1.1G的lzo压缩文件，那么处理第二个128MBblock的mapper就必须能够确认下一个block的boundary，以便进行解压操作。lzo并没有写什么数据头来做到这一点，而是实现了一个lzoindex文件，将这个文件（foo.lzo.index）写在每个foo.lzo文件中。这个index文件只是简单的包含了每个block在数据中的offset，这样由于offset已知的缘故，对数据的读写就变得非常的快。通常能达到90-100MB/秒，也就是10-12秒就能读完一个GB的文件。一旦该index文件被创建，任何基于lzo的压缩文件就能通过load该index文件而进行相应的分块，并且一个block接一个block的被读取。也因此，各个mapper都能够得到正确的block，这就是说，可以只需要进行一个LzopInputStream的封装，就可以在hadoop的mapreduce中并行高效的使用lzo。如果现在有一个job的InputFormat是TextInputFormat，那么就可以用lzop来压缩文件，确保它正确的创建了index，将TextInputFormat换成LzoTextInputFormat，然后job就能像以前一样正确的运行，并且更加的快。有时候，一个大的文件被lzo压缩过之后，甚至都不用分块就能被单个mapper高效的处理了。
在hadoop集群中安装lzo
要在hadoop中搭建lzo使用环境非常简单：    

1.	安装lzop native libraries
例如：```sudo yum install lzop lzo2```

2.	从如下地址下载 hadooplzo支持到源代码：http://github.com/kevinweil/hadoop-lzo
3.	编译从以上链接checkout下来到代码，通常为：ant compile-native tar
4.	将编译出来到hadoop-lzo-*.jar部署到hadoop集群到各个slave到某个有效目录下，如$HADOOOP_HOME/lib
5.	将以上编译所得到hadoop-lzo native libbinary部署到集群到某个有效目录下，如$HADOOP_HOME/lib/native/Linux-amd64-64。
6.	将如下配置到 core-site.xml 中：    

```
	<property>
	<name>io.compression.codecs</name>
	<value>org.apache.hadoop.io.compress.GzipCodec,org.apache.hadoop.io.compress.DefaultCodec,org.apache.hadoop.io.compress.BZip2Codec,com.hadoop.compression.lzo.LzoCodec,com.hadoop.compression.lzo.LzopCodec</value>
	</property>
	<property>
	<name>io.compression.codec.lzo.class</name>
	<value>com.hadoop.compression.lzo.LzoCodec</value>
	</property>    
```

7. 将如下配置到mapred-site.xml中：        
```
	<property>
     <name>mapred.child.env</name>
	<value>JAVA_LIBRARY_PATH=/path/to/your/native/hadoop-lzo/libs</value>
	</property>
	如果想要mapreduce再写中间结果时也使用压缩，可以将如下配置也写入到mapred-site.xml中。
	<property>
	<name>mapred.map.output.compression.codec</name>
	<value>com.hadoop.compression.lzo.LzoCodec</value>
	</property>
```    

如果以上所有操作都成功，那么现在就可以尝试使用lzo了。比如打包一个lzo都压缩文件，如lzo_log文件，上传到hdfs中，然后用以下命令进行测试：    
```
hadoop jar /path/to/hadoop-lzo.jarcom.hadoop.compression.lzo.LzoIndexerhdfs://namenode:9000/lzo_logs
```    

如果要写一个job来使用lzo，可以找一个job，例如wordcount，将当中到TextInputFormat修改为LzoTextInputForma，其他都不用修改，job就能从hdfs上读入lzo压缩文件，进行分布式都分块并行处理。


>Using Hadoop and LZO
Reading and Writing LZO Data
The project provides LzoInputStream and LzoOutputStream wrapping regular streams, to allow you to easily read and write compressed LZO data.
Indexing LZO Files
At this point, you should also be able to use the indexer to index lzo files in Hadoop (recall: this makes them splittable, so that they can be analyzed in parallel in a mapreduce job). Imagine that big_file.lzo is a 1 GB LZO file. You have two options:
•	index it in-process via:
•	hadoop jar /path/to/your/hadoop-lzo.jar com.hadoop.compression.lzo.LzoIndexer big_file.lzo
•	index it in a map-reduce job via:
•	hadoop jar /path/to/your/hadoop-lzo.jar com.hadoop.compression.lzo.DistributedLzoIndexer big_file.lzo
Either way, after 10-20 seconds there will be a file named big_file.lzo.index. The newly-created index file tells the LzoTextInputFormat's getSplits function how to break the LZO file into splits that can be decompressed and processed in parallel. Alternatively, if you specify a directory instead of a filename, both indexers will recursively walk the directory structure looking for .lzo files, indexing any that do not already have corresponding .lzo.index files.
Running MR Jobs over Indexed Files
Now run any job, say wordcount, over the new file. In Java-based M/R jobs, just replace any uses of TextInputFormat by LzoTextInputFormat. In streaming jobs, add "-inputformat com.hadoop.mapred.DeprecatedLzoTextInputFormat" (streaming still uses the old APIs, and needs a class that inherits from org.apache.hadoop.mapred.InputFormat). Note that to use the DeprecatedLzoTextInputFormat properly with hadoop-streaming, you should also set the jobconf propertystream.map.input.ignoreKey=true. That will replicate the behavior of the default TextInputFormat by stripping off the byte offset keys from the input lines that get piped to the mapper process. For Pig jobs, email me or check the pig list -- I have custom LZO loader classes that work but are not (yet) contributed back.
Note that if you forget to index an .lzo file, the job will work but will process the entire file in a single split, which will be less efficient.

参考资料    

[lzo本地压缩与解压缩实例](http://blog.csdn.net/scorpiohjx2/article/details/18423529)    
[hadoop集群内lzo的安装与配置](http://share.blog.51cto.com/278008/549393/)     
[安装 Hadoop 2.0.0-cdh4.3.0 LZO 成功](http://www.tuicool.com/articles/VVj6rm)     
[hadoop-lzo源代码](https://code.google.com/a/apache-extras.org/p/hadoop-gpl-compression/wiki/FAQ?redir=1)     
[Hadoop Could not load native gpl library异常解决](http://guoyunsky.iteye.com/blog/1237327)
