---
layout: post
title: 论文:在线游戏玩家坐标聚类分析
description: 游戏安全论文
category: 游戏安全
mathjax: true
---

### 1 玩家聚类

首先，我们提出一个方法来检测坐标，然后描述如何使用MDS(multidimensional scaling)和一个称为prefuse的方法计算坐标之间的转移权重概率(weighted transition probabilities between landmarks(WTPL))，这些用于可视化玩家聚类。研究证明，MDS在多维数据集聚类时中要优于SOM。在prefuse阶段，我们使用一种等价于MDS的力导向(force-directed)的模型。文中的**坐标点(Landmark)** 与普通的定义不同。

#### 1.1 坐标(landmark)检测

首先，要分析的地图划分为$m\times n$的网格，假设有N个玩家遍历地图。其中用户经过网格$$(i,j)$$的数目标识为 $v_{i,k}(k)$，我们将大部分玩家都高频率经过的网格定义为坐标。对于网格 $(i,j)$ ,玩家经过的分布可以用加权熵H来表示
$$
H_{i,j}=-\frac{V_{i,j}}{\hat V}\sum _{k=1} ^N\frac{v_{i,j}(k)}{V_{i,j}}log(\frac{v_{i,j}(k)}{V_{i,j}}),其中V_{i,j}=\sum _{k=1} ^Nv_{i,j}(k)\quad ,\hat V = argmax_{i,j}V_{i,j}
$$ 

检测坐标L的算法如下
1. 所有的网格为未标记状态，所有坐标都是空
2. 在所有未标记网格中，将有最大H的网格标记为1，并将此网格加入到坐标集中。然后将其8个邻居也标记为1。 此处标记邻居是防止被连续标记。
3. 重复步骤2直至坐标集达到L个

#### 1.2 坐标之间的加权转移概率

对于上面算法检测到的L个坐标，用户k的加权概率转移矩阵$X(k)$如下
$$
X(k) =
\left[\begin{array}{cccc}
p_{1,1}(k) & p_{1,2}(k) & ... p_{1,L}(k) \\
p_{2,1}(k) & p_{2,2}(k) & ... p_{2,L}(k) \\
... & ... & ...  \\
p_{L,1}(k) & p_{L,2}(k) & ... p_{L,L}(k)
\end{array}\right]
$$
上式中，$P_{a,b}(k)$为玩家k从坐标a移动到坐标b的加权转移概率。其累加公式为
$$
P_{a,b}(k)=\frac{c_{a,b}(k)}{\hat c_{a,b}}\frac{c_{a,b}(k)}{C(k)}\quad 其中c_{a,b}(k)为玩家从坐标a移动到坐标b的次数,\hat c_{a,b}(k)=argmax_kc_{a,b}(k)\quad C(k)=\sum _{i=1} ^L\sum _{j=1} ^Lc_{i,k}(k)
$$。

当前版本中为计算$X(k)$中各个元素值，玩家所在位置(coordinate)由其轨迹的附近坐标(landmark)来表示。假设首先已经检测到5个坐标$(A,B,C,D,E)$，那么玩家的轨迹可能是$BBBAAAACEEEE$，其转移序列为$B\rightarrow A\rightarrow C\rightarrow E$。当前不考虑自循环转移序列，如$A\rightarrow A,B\rightarrow B$

#### 1.3 Multildimensional Scal(MDS(多维缩放))

为可视化基于其运动模式的玩家聚类，我们使用简单的分类MDS(CMDS(论文1))。CMDS的输入为矩阵D，给定一对玩家的差异性，然后输出一个坐标矩阵，矩阵以配置了最小损失函数来保留两点之间的距离。当前论文中矩阵D中的第$ij$个元素为$X(i)和X(j)$之间的欧氏距离。我们只选取结构坐标的前两个维度，并使用Matlab中的统计工具包中的$cmdscale$函数来计算CMDS。

#### 1.4 prefuse(预测)
我们同时在预测阶段中使用一个力导向的模型，其输出一个图分布，图中每个节点代表一个玩家，节点之间的连接长度与两个节点间距离成正比。为减少计算复杂度，我们提前删除节点间距离小于一定阈值的，当前可以设定为所有节点距离中值的0.5.在预测阶段用户可以交互地操作图节点布局来定位聚类簇

### 2 结果与讨论
使用的是游戏ICE的玩家移动日志数据
#### 2.1 游戏ICE

游戏截图如下左，分析所使用的是图右（图中有7个固定位置的NPC），图尺寸为$600\times 600$像素。玩家可以通过点击目标点像素来移动，当前地图被划分为$50\times 50$的网格。

![game_bot_detection_0](/images/blog/paper_onlinegame_player_landmark_cluster_0.jpg)
![game_bot_detection_0](/images/blog/paper_onlinegame_player_landmark_cluster_1.jpg)

下图展示的是当$L=5$时，检测坐标(Landmark)的结果。坐标1：靠近Marlen和Lagi；坐标2：所有玩家进入地图的位置；坐标3：Gelda所在位置；坐标4：位于Amanda和Villege Master之间；坐标5：同时靠近Shop和Gelda。这些坐标对所有玩家都是重要的，因为他们要进入地图、接收任务、购买物品。

![game_bot_detection_0](/images/blog/paper_onlinegame_player_landmark_cluster_2.jpg)

有20个玩家，17个男性，3个女性游戏时长45分钟，完成一些简单的任务，如与NPC交易物品、运送物品到NPC，从怪物那收集物品。下图画出了所有玩家在二维空间中由CMDS检测出来的结果，节点$p_i$代表第$i$个玩家。注意有个离群点$p_{11}$，因为所有玩家都在右边，我们剔除掉此玩家然后重新计算结果得到下图右边。
![game_bot_detection_0](/images/blog/paper_onlinegame_player_landmark_cluster_3.jpg)
![game_bot_detection_0](/images/blog/paper_onlinegame_player_landmark_cluster_4.jpg)

上图右可以看出，大部分玩家可以被划分到4个聚类($A,B,C,D$)中.每个聚类都有不同的运动模式。每个聚类的运动模式由簇中每个元素的平均的$X$表示，并由一个与$X$的值成正比的箭头可视化。下图展示了这种可视化结果，聚类A中的玩家运动模式主要是在坐标2到5和3到5之间移动。聚类簇B中的玩家主要活动模式是2到4和1到2之间移动；聚类簇C中的玩家是在3到5和2到5，聚类簇D中玩家主要是1到2和2到5之间。

![game_bot_detection_0](/images/blog/paper_onlinegame_player_landmark_cluster_5.jpg)
![game_bot_detection_0](/images/blog/paper_onlinegame_player_landmark_cluster_6.jpg)

#### 2.2 游戏：天使之爱(Angle’s Love)

下图7是游戏截图和对应的检测到的5个坐标点。图尺寸为$300\times 190$像素，图被划分为$10\times 10$网格。由397个玩家在此图中游戏，大部分从此图进入其他地图。这5个坐标中，其中的3个即1，2，3是障碍物，玩家遇到之后必须返回。这也解释了为什么它们会被检测为坐标点(Landmark)，因为玩家要在这三个点之间来回两次，过去、返回。

![game_bot_detection_0](/images/blog/paper_onlinegame_player_landmark_cluster_7.jpg)

下图展示了预测结果的图布局。图中形成了一个大的聚类簇，表明大部分玩家都有类似的运动模式，但是也可以看到顶部和底部各有一个3个玩家和4个玩家组成的小得多得聚类。

![game_bot_detection_0](/images/blog/paper_onlinegame_player_landmark_cluster_8.jpg)

进一步查看这两个小聚类簇，下图左边是顶部的三个玩家在坐标点之间的移动轨迹，右边是底部4个玩家坐标点之间的移动轨迹，可以看到它们的移动轨迹相似。

![game_bot_detection_0](/images/blog/paper_onlinegame_player_landmark_cluster_9.jpg) ![game_bot_detection_0](/images/blog/paper_onlinegame_player_landmark_cluster_10.jpg)


#### 论文

1. D. Ashbrook and T. Starner. Using GPS to learn significant locations and predict movement across multiple users, Personal and Ubiquitous Computing,









