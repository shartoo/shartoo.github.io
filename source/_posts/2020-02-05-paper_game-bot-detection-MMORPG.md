---
layout: post
title: l论文:MMORPG游戏中人机检测方法总结
description: 游戏安全论文
category: 游戏安全
mathjax: true
---

### 0 背景

+ 此论文整理的是: Ah Reum Kang1, Seong Hoon Jeong2, Aziz Mohaisen1 and Huy Kang Kim :Multimodal game bot detection using user behavioral characteristics

游戏中人机检测方法可以分为三类。

+ 客户端：大部分公司采用客户端，分析人机与普通玩家的显著区别。可以使用人机程序的名字，处理器信息，内存状态。工作原理类似于杀毒程序。但是此类方法很容易被人机程序开发者避开，所以此方法并非最优
+ 网络端：`网络通信监督`,`网络协议改变分析`,这些方法会导致网络过载或者游戏延迟，给游戏体验带来不好。
+ 服务端：主要是分析游戏日志数据，基于数据挖掘方法。游戏公司可以在服务端直接检测人机，不必在客户端部署程序。

下表整理了一些服务端人机检测的关键论文，并做了一些分类。

+ **动作频率分析**：准确率高，但是只聚焦于短时窗口的观察，很容易被规避；只集中于有限特征空间，容易误分人机和重度玩家
  - [User identification based on game-play activity patterns](https://www.researchgate.net/publication/221391430_User_identification_based_on_game-play_activity_patterns)玩家特定动作的动态性，证明了玩家的活跃和休息时间可以区分人机
  - [Detection of MMORPG misconducts based on action frequencies, types and timeintervals](https://www.researchgate.net/publication/220705240_Detection_of_MMORPG_Misconducts_Based_on_Action_Frequencies_Types_and_Time-Intervals)使用日志数据中动作频率、类别、间隔
  - [Game behavior pattern modeling for game bots detection in MMORPG](https://arxiv.org/ftp/arxiv/papers/1509/1509.02458.pdf):选取六种游戏特征。`换地图`,`counter-turn（逆转）`,`休息状态`,`击杀时间`,`经验点`,`留在城镇内`。
  - [Game bot detection approach based on behavior analy‑ sis and consideration of various play styles.](https://onlinelibrary.wiley.com/doi/full/10.4218/etrij.13.2013.0049)根据玩家游戏风格划分为4种玩家类型：`杀手`，`成就者`,`探险者`,`社交爱好者`。
  - [Detection of illegal players in massively multiplayer online role playing game by classification algorithms](https://www.researchgate.net/publication/308850232_Detection_of_Illegal_Players_in_Massively_Multiplayer_Online_Role_Playing_Game_by_Classification_Algorithms)根据游戏时间划分游戏行为
+ **社交活动分析**：无法检测游戏中不当行为
  - [Second life: a social network of humans and bots](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.165.4976&rep=rep1&type=pdf):聚焦于玩家社交网络中玩家之间连接分析
  - [Chatting pattern based game bot detection: do they talk like us?](https://www.researchgate.net/publication/275653414_Chatting_pattern_based_game_BOT_detection_Do_they_talk_like_us):聊天日志分析玩家沟通模式。
  - [Bot detection based on social interactions in MMORPGs](https://www.researchgate.net/publication/261310404_Bot_detection_based_on_social_interactions_in_MMORPGs) 人机和人类倾向于各自相反的自有社交网络。
  - [Online game bot detection based on party‑play log analysis](https://www.sciencedirect.com/science/article/pii/S0898122112000442) 人机和玩家组团游戏的目标不同。
+ **掘金(Gold Farming group)组分析**：分析游戏中虚拟经济，并追踪由金币农民、中间商、币商(banker)、买家形成的交易网络。尽管不是直接分析人机，但是有助于理解网络中每个玩家的角色。
  - [Exploiting MMORPG log data toward efficient rmt player detection](https://www.researchgate.net/publication/220982695_Exploiting_MMORPG_log_data_toward_efficient_RMT_player_detection):分析四种统计数据,`总动作数`,`活动时间(activity time)`,`总聊天次数`,`一段时间内游戏内总货币流通量`。
  - [Detecting gold‑farmers’ groups in MMORPG by connection information](https://www.researchgate.net/publication/264033427_Detecting_gold-farmers'_group_in_MMORPG_by_analyzing_connection_pattern)使用路由和源定位信息分析了金币农民的连接模式。
  - [Surgical strike: a novel approach to minimize collateral damage to game bot detection.](https://www.researchgate.net/publication/269305755_Surgical_strike_A_novel_approach_to_minimize_collateral_damage_to_game_BOT_detection)调查了金币农民网络并检测到整个金币农民组的网络结构
+ **序列分析**：分析用户游戏中动作序列，但是因为分析上欠缺场景上下文，很容易被人机设置操纵。
  - [Mining for gold farmers: automatic detection of deviant players in MMOGs](https://www.researchgate.net/publication/220775532_Mining_for_Gold_Farmers_Automatic_Detection_of_Deviant_Players_in_MMOGs)研究了活动序列特征，定义玩家参与某个活动的次数，玩家击杀怪物次数和被怪物击杀次数。
  - [Sequence‑based bot detection in massive multiplayer online games](https://sci-hub.tw/10.1109/ICICS.2011.6174239)分析玩家在游戏中战斗(combat)序列
  - [In‑game action sequence analysis for game bot detection on the big data analysis
platform](https://www.researchgate.net/publication/275653312_In-Game_Action_Sequence_Analysis_for_Game_BOT_Detection_on_the_Big_Data_Analysis_Platform)基于大数据平台分析了用户的全部动作序列。
+ **相似度分析**：基于一个事实，即人机有强烈的行为模式因为它们的目标是赚更多金币，需要大量的特定行为动作的数据。
  - [Self‑similarity based bot detection system in MMORPG](https://www.researchgate.net/publication/275653327_Self-similarity_based_Bot_Detection_System_in_MMORPG)使用每个事件的频率(量化为向量)并计算与单位向量的的余弦相似度，人机会重复做高度重复的动作序列，所以其动作序列有非常高的自相似度。
  - [You are a game bot!: uncovering game bots in MMORPGs via self-similarity in the wild.](https://pdfs.semanticscholar.org/444f/0ceb312e98609914faf886f2ff0dcfebd58c.pdf?_ga=2.54103150.790611450.1582517402-1588539036.1582517402)提出了自相似度衡量方法并在几个主要MMOPRG游戏("Lineage","AION","Blade & Soul")中测试
+ **移动路径分析**：人机有特定的移动路径，而人类则有各种各样的路径。此类方法容易被噪音干扰或者被类人的行为模式规避。
  - [Detection of landmarks for clustering of online‑game players](https://www.researchgate.net/publication/220222180_Detection_of_Landmarks_for_Clustering_of_Online-Game_Players)使用玩家角色在游戏地图中遍历位置的分布的加权熵来检测其坐标(landmark),基于坐标转移概率做了玩家聚类。
  - [A step in the right direction: bot detection in MMORPGs using movement analysis](https://www.researchgate.net/publication/290512402_A_step_in_the_right_direction_Botdetection_in_MMORPGs_using_movement_analysis)借由人机和人类移动模式的差异进行分类。
  - [Server‑side bot detection in massively multiplayer online games](http://www.eurecom.fr/en/publication/2860/download/rs-publi-2860.pdf)基于重复的运动模式检测出由脚本控制的人机。
  - [Game bot detection via avatar trajectory analysis](https://www.iis.sinica.edu.tw/~swc/pub/bot_detection_trajectory.html)使用玩家路径和一系列位置坐标的熵值，使用马尔科夫链模型来描述目标路径的行为。
  - [Trajectory analysis for user verification and recognition](https://www.researchgate.net/publication/262293070_Trajectory_analysis_for_user_verifcation_and_recognition)将它们的方法(上面这一篇马尔科夫链的论文)应用于各种各样的路径，包括手写字、鼠标、游戏路径

### 1  数据集

使用的是现实世界中游戏Aion的操作数据

+ 类型：游戏中动作日志
+ 持续事件：88天。 0.4/09/2010-07/05/2010
+ 角色数量：49739
+ 角色最小游戏时间: >3小时

### 2 分析架构和工作流

将游戏中人机识别当作一个**二分类任务**。总体流程如下

![game_bot_detection_0](/images/blog/game_bot_detection_0.png) 
 
+ **数据采集**： 从游戏日志和聊天内容中收集
+ **数据探索**
  - 在特征表征阶段，我们遵循标准方法来统一数据来降维。例如，量化每个网络并使用k-means聚类成低、中、高三个不同值。
  - 在特征探索阶段，我们选取数据向量的组件并预处理。例如，我们决定了7种活动为社交互动，并使用香农多样性熵来量化社交互动的多样性。
  - 特征选取阶段，我们使用best-first搜索逐步贪婪搜索以及信息增益排序过滤来避免过拟合并减少特征的方法来选取显著特征。

+ **机器学习**：选取算法（决策树、随机森林、逻辑回归和朴素贝叶斯等）和参数（k折交叉验证，特定算法参数等）
+ **评估**：根据公司提供的已经被划入黑名单的账户列表来统计每个分类器的性能。统计精确度、召回率、F-score等。
+ **已用特征和它们的差距**：如下表，我们将使用过的特征分类为`个人`和`社交`。鉴于人机的目的是赚取更多利润，其个人特征与人类用户有区别。

+ 个人特征
  - 玩家信息：登录频率，游戏时间，游戏金钱，IP地址数目
  - 玩家动作：端坐(玩家用以回复生命状态的动作（人机端坐更频繁以恢复生命值）)，挣经验点，获得物品项(其他玩家赠送)，挣游戏金币，挣玩家击杀点(人类玩家击杀越多，排名越高，人机不在乎排名)，收获物品(击杀boss掉落)，复活，restore经验点，被非玩家或非玩家角色击杀，使用门户(using portals)
+ 社交特征(人机不热衷社交)
 - 群组活动：组团游戏(人机可能也组团，但是他们更在乎高效打钱和掉落物品，人类玩家更在乎完成任务)，工会活动
 - 社交互动多样性：组团游戏，游戏，交易，私信，邮件，商铺，工会。人类为了在线游戏会执行多种任务，而人机只会集中在几个固定动作。
 - 网络度量：节点中心度(Degree centrality(与此节点连接的数目))，中间性(betweenness centrality(两个节点之间的最短路径)),亲密度(某节点与与其他所有节点越近代表越重要)，特征向量中心性(eigenvector centrality(好几个邻居节点的特征向量中心性值很高)),偏心度(eccentricity),authority(被很多好的hub指向的节点),hub（指向很多好的authority的节点）,PageRank,聚集系数(clustering coefficient(团体聚合程度))

### 3 结果和讨论

+ 行为特征
  - 玩家信息：人机会24小时一直在线，并且会在上班时间也在线。下图展示了人机是多么密集地玩游戏。下图图c展示了用户每天收获物品项地最大值地累计分布，用户每天要收获超过1000个物品基本不可能，但是60%的人机每天收获物品超过5000个。这是一个显著特征
  - ![game_bot_detection_0](/images/blog/game_bot_detection_1.png) 
  - 玩家动作：下图展示了人机/人类玩家的活动比率,红色点代表人机，蓝色点代表人类。人机挣游戏币的比率基本接近人类。但是人机挣取经验点和获取物品项比率远超人类。人机的`挣取经验点`,`获得物品项`,`挣取游戏币`的累计比率都是0.5，而人类只有0.33。这表明人机集中于利润相关活动，而人类享受 各种各样的活动。相反的，人类玩家的击杀玩家点是人机的三倍。人机不太在乎排名。
  - ![game_bot_detection_0](/images/blog/game_bot_detection_2.png) 

+ 组团活动
  - 下图展示了人机和人类玩家的平均组团时间。人机组团的游戏模式比较异常，人机中间会配合不被怪物杀死。**80%的人机组团时间超过4h10min，而80%人类组团不超过2h20min**。即便异常困难的组团，人类时间也不如人机。
  - ![game_bot_detection_0](/images/blog/game_bot_detection_3.png) 
+ 社交多样性
  - 下图展示了社交熵的累计分布。被定义为社交活动的有：组团，合伙，交易，私信，邮件，商店，工会。量化社交活动多样的方法是香浓多样性熵 $H^=-\sum _{i=1} ^n p_iln p_i\quad n为社交活动类型,p_i为第i个社交活动类型的相对比例$ 。人类玩家比人机更享受多种活动。
  - ![game_bot_detection_0](/images/blog/game_bot_detection_4.png) 
+ 网络度量
  - 下表分析了人机和人类玩家的组团行为。首先，我们可以看到玩家网络中**人类组团的平均度是人机的18倍**。因为人类玩家网络中会存在很多不认识的，而人机只跟特定其他人机玩。**人类团体的网络友谊度差不多是人机的4倍**。**交易网络中人类玩家的平均度是人机的2.5倍**。但是，人机网络的cluster coeffcient聚合系数是人类玩家的5倍。在邮件网络中，我们也看到人机之间会发送一些垃圾邮件，同时也注意到存在5个收集者从其他人机那里搜集到6000个物品。这表明存在其中存在掘金团体。同时，由于在商人模式下，玩家无法移动，所以人机很少处于商人模式。
  - ![game_bot_detection_0](/images/blog/game_bot_detection_5.png) 

+ 三角调查
  - 下图中的13个三角网络主题中的相对问题表明网络中更细节的互动模式。对于当前的Aion游戏网络，在图b以每个主题类型的比例和与空模型对比的Z-score($Z_i = \frac{N_i ^{real}-N_i ^{random}}{\sigma _i ^{random}}\quad 其中N_i ^{real}是真实观测到的主题i数目，N_i ^{random}是随机网络中的期望值,\sigma _i ^{random}是随机网络中主题数期望值的标准差$)分值展示了互动模式。
  - **结论**：我们发现，人机的好友网络、私信、邮件、商店网络，以及人类玩家的游戏和商店网络表现出一中主导类型的主题类型。例如，好友网络，类型7占据了90%的三元组关系，这充分展现了互动的倒数性质。相反的性质可以用于商店，低倒数反映出存在大商家。在人机的私信和邮件网络中，类型1的账户占据了80%的节点三元组关系。这表明，在私信网络中一些人机发送怪物的坐标信息给其他人机。
  - ![game_bot_detection_0](/images/blog/game_bot_detection_6.png) 
  - 将主题的缺失与空模型的对比使得我们能够检测到由于随机值造成的损耗，这个可以通过Z-score值完成。这在其他两个网络(组团party和交易网络)也很有必要，将空网络考虑在内的时候，我们可以发现尽管多种主题可能是相似的丰富（上图b），但是一些可能是十分显著或者未被表达出来。在人类团体中，过度表达的主题类型5($\hat Z>0.4，标准化的版本\hat Z=Z_i\sqrt{\sum_i(Z_i ^2)}$)实际上十分接近于三角形，这与组团party网络中的高聚合度趋势一致。在人机组团中，过度表示的类型13表明网络中观测到的主题数目与在随机网络中出现的主题数目的期望值之间存在巨大差距。这表明，人机内部存在自己的组团来帮助彼此并交易。
  - ![game_bot_detection_0](/images/blog/game_bot_detection_7.png) 

+ 网络重叠
  - 为研究配对网络之间的相关度，我们研究了人机和人类组团之间的网络相似度。例如，两个网络可能展示出相似的聚类值，但这并不能保证与某节点相连的其他节点在另外一个网络也与其相连，或者节点表现出相似的活跃度。有两种方式来衡量网络的重叠度。第一个是**雅可比相似度**，第二个是**网络对(pairs)**的节点度的**Pearson相关系数**。下图是人机和人类组团的10个网络对(pairs)的链接和度重叠的结果。通过链接重叠（图a），我们发现人机组团在party-friendship(组团-好友)网络对以及party-trade(组团-交易)网络对中有更高的雅可比相似度。这是因为人机的主要活动是组团游戏和交易物品。图b中节点度重叠是另外一种发现互动的关联。例如，人类团体的party-trade(组团-交易)网络表现出大于0.7的正的Pearson相关系数，这可以理解为人类的组团活动，例如在团战或者狩猎时需要交易物品。相反的，人机网络中的Pearson系数就非常低。
  - ![game_bot_detection_0](/images/blog/game_bot_detection_8.png) 


### 4 检测方法详述

+ 使用了一种**判别式方法**来学习人机和人类玩家的特征差异。使用了**10-折交叉验证**,将数据分为10份
+ 特征选取：我们对比了模型结果与公司提供的黑名单来评估选取的特征。进行特征选取的步骤是先使用best-first，逐步贪婪，以及信息增益排名过滤算法类提升选取过程。下图显示了使用了如下三种特征的分类算法的结果
 - Feature_Set1包含了上面`方法`小节提到的全部的特征114个
 - Feature_Set2是由信息增益过滤器算法抽取的前62个特征,结果与Feature_Set1接近，但是特征数少很多。所以选取此特征集。
 - Feature_Set3 是由best-first和逐步贪婪算法组合选取的6个特征。结果不如上面两个
 - ![game_bot_detection_0](/images/blog/game_bot_detection_9.png) 
+ 分类与评估：人机的行为模式检测结果如下表。四种分类器用于训练算法，`决策树`，`随机森林`，`逻辑回归`，`朴素贝叶斯`。使用的都是Feature_Set2的特征。其中随机森林表现最好。训练数据集中有85%人类玩家和15%的人机玩家，随机森林算法能应对数据不均衡
  - ![game_bot_detection_0](/images/blog/game_bot_detection_10.png) 
  - 尽管我们进行了特征选取，但是还是有很多相关特征。例如，获取物品数目，挣取经验点数目，最大收获物品数目，组团偏心性(party_eccentricity) ，游戏时间，获得物品比例都是不显著的特征。尽管这些特征不显著，但是他们天然相关，不可以简单地剔除掉。
  - 下表显示了分别类评估结果的相对相似性和差异性：TP,FP,FN,TN。为获得相对相似性，我们通过最低分类数目标准化所有分类，来比较相对结果。对除去最低分类的每个分类，计算其除以最低分类得到的比例。相对相似度的模式与大部分特征和分类是一致的，除了`mail_between_centrality(邮件网络的中介性)`和`mail_outdegree(邮件网络的节点出度)`
 - ![game_bot_detection_0](/images/blog/game_bot_detection_11.png) 



#### 其他数据

![game_bot_detection_0](/images/blog/game_bot_detection_12.png) 
![game_bot_detection_0](/images/blog/game_bot_detection_13.png) 
![game_bot_detection_0](/images/blog/game_bot_detection_14.png) 












