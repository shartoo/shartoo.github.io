---
layout: post
title: 论文:游戏中人机检测2017年之后论文概述
description: 游戏安全论文
category: 游戏安全
mathjax: true
---

### 0 相关网站

1. [魔兽世界中国区道具买卖](https://www.g2g.com/wow-us/Item-2299-19260)
2. [上千种特征丰富的人机论坛](https://www.raccoonbot.com/)
3. [人机使用心得交流论坛](https://mybot.run/forums/)
### 1 引言
### 1.1 人机检测的逻辑和方法

1. 文献搜索：可以搜索的关键词是`game bot detection`,`game bot security`,`game bot cheating detection`。分析了从谷歌学术搜索三个关键词各返回的前20篇论文。只保留2017年之后的论文，剩下35篇。
2. 文献筛选：选取标准是根据 [JUFO](https://www.tsv.fi/julkaisufoorumi/haku.php?lang=en) 文章评分，0-3分，得分越高，质量越好。经过这次筛选，我们只得到12篇论文。

论文列表如下

|文章(大部分是ResearchGate)|分析的数据类别|游戏类别|数据集|
|---|---|---|---|
|[Game Bot detection based on avatar trajectory.](https://www.researchgate.net/publication/220851327_Game_Bot_Detection_Based_on_Avatar_Trajectory)|角色轨迹|FPS(雷神之锤2)|在与游戏相关的5个网站公开|
|[Multimodal Game Bot Detection using User Behavioral Characteristics](https://www.researchgate.net/publication/301640460_Multimodal_Game_Bot_Detection_using_User_Behavioral_Characteristics)|动作频率和社交活动|MMORPG(AION)|公开的AION数据集|
|[Online game bot detection based on party-play log analysis](https://www.researchgate.net/publication/257313135_Online_game_bot_detection_based_on_party-play_log_analysis)|动作频率和社交活动|MMORPG(AION)|私有的AION数据集|
|[Server-side bot detection in massively multiplayer online games](https://www.researchgate.net/publication/220496922_Server-Side_Bot_Detection_in_Massively_Multiplayer_Online_Games)|角色轨迹|MMORPG(魔兽世界)|WOW私有数据集|
|[A time series classification approach to game bot detection](https://www.researchgate.net/publication/318717059_A_time_series_classification_approach_to_game_bot_detection)|动作频率和社交活动|MMORPG(AION)|公开的AION数据集|
|[Battle of Botcraft: fighting bots in online games with human observational proofs](https://www.researchgate.net/publication/221609548_Battle_of_Botcraft_Fighting_bots_in_online_games_with_human_observational_proofs)|动作频率|MMORPG(魔兽世界WOW)|私有数据集(鼠标和键盘事件)|
|[Behavioral-based cheating detection in online first person shooters using machine learning techniques](https://sci-hub.tw/10.1109/cig.2013.6633617)|动作频率|FPS(特洛伊战争)|私有数据集(游戏事件)|
|[Chatting pattern based game BOT detection: do they talk like us?](https://www.researchgate.net/publication/275653414_Chatting_pattern_based_game_BOT_detection_Do_they_talk_like_us)|社交活动|MMORPG(AION)|私有数据集（游戏事件和聊天活动）|
|[Crime scene reconstruction: online gold farming network analysis](https://sci-hub.tw/10.1109/TIFS.2016.2623586)|社交活动和网络边(network-side)|MMORPG(AION)|私有数据集(交易日志)|
|[Detection of MMORPG bots based on behavior analysis](https://sci-hub.tw/10.1145/1501750.1501770)|动作频率|MMORPG([Cabal Online](https://store.steampowered.com/app/253490/CABAL_Online/))|事件日志|
|[NGUARD: A Game Bot Detection Framework for NetEase MMORPGs](https://www.researchgate.net/publication/326503248_NGUARD_A_Game_Bot_Detection_Framework_for_NetEase_MMORPGs)|动作频率|网易游戏MMORPG|私有数据集(动作日志)|
|[Game Bot detection approach based on behavior analysis and consideration of various play styles](https://arxiv.org/abs/1509.02458)|动作频率|MMORPG(热血江湖Yulgang Online)|私有数据集(事件日志)|

#### 1.2  研究的问题

本文研究如下问题
1. 哪类游戏的人机检测是科研社区关注度最低的。就是看看哪类游戏还有机会做出新的突破
2. 人机检测有哪些公开数据集。可以用于实验
3. 如何检测在线游戏。梳理人机检测的可靠方法
4. 人机检测的未来方向

### 2 研究的游戏类别

类别之间可能相关，并且一个游戏也可能同时属于多个类别。有如下类别

+ **角色扮演**：玩家控制游戏中角色探索虚拟世界。此类游戏一般是玩家搜集物资来增强角色能力，其中一个子类是MMOPRG，例如魔兽世界。
+ **动作游戏**：玩家控制角色摧毁或者躲避游戏中障碍物，典型游戏是超级玛丽。
+ **冒险游戏**：控制角色解谜题类任务，角色任务相对固定。此类别游戏集中于故事线，例如`Syberia`
+ **策略游戏**：玩家控制角色的人口并替这些角色做决定，比如星际争霸
+ **音乐游戏**：玩家需要完成音乐曲目或者跳舞来获得经验点，例如吉他英雄。
+ **射击游戏**：控制角色击杀其他玩家，一般是使用枪。典型的如FPS，如雷神之锤
+ **格斗游戏**：玩家控制角色与其他角色格斗，如真人快打
+ **解谜游戏**：玩家不需要控制任何角色，只需要解出游戏中各种谜题即可
+ **赌博游戏**：如轮盘赌博

大部分论文集中于MMORPG和FPS游戏中的人机检测。但是一些其他策略手游如皇室战争(Clash Royale)有2700万玩家，有些人使用人机提升角色并在后续卖掉，这类并没有获得很高关注度。

### 3 数据集

下面列出一些公开的数据集，虽然研究者用了各种各样的数据集，但是大部分都无法公开访问。

+ **Quake 2(雷神之锤2)数据集**：游戏允许记录游戏日志，可以后面用以观看和分析。日志包括角色移动，和游戏事件如捡起物品、射击和摧毁角色。玩家分享了一些游戏日志，如[Planet Quake](http://q2scene.net/ds/)和[Demo Squad](http://ocslab.hksecurity.net/Datasets/game-bot-detection)。
+ **AION数据集**：数据集包含了49739个角色2010年4月9日到7月5号的动作日志，同时数据集包含一些人工排查被封号的用户列表。

如果说一些公开数据据不符合研究者需要，它们就会自己按照如下思路构建一些数据集。
+ **追踪事件**：在游戏客户端可以追踪一些事件，如点击和键盘按键。
+ **开发一个游戏**：研究者可能自己开发一个游戏来研究并追踪游戏中的事件。
+ **数据集生成**：为了生成人机网络数据集，一些论文使用了分析真实数据包追踪来构建数据并生成真实数据集。通过探索多阶段场景生成了大量事务数据。


### 4 人机检测方法

#### 4.1 无监督和有监督学习方法

如论文1，2，3，4所述方法，这些监督方法需要有标注的数据集，用户被标记为人机或者普通用户，使用这样的数据集就可以训练分类模型。
无监督学习方法不需要标注，典型的如异常检测算法用于检测人机的异常行为，也可以使用聚类算法。

#### 4.2 基于用户行为的分类方法

基于用户行为的分类方法分为三种

+ 客户端的：有点类似于反病毒的，安装在客户端的程序，程序会检测用户是否有人机行为
+ 网络端(network-side)的：基于网络中收发的数据包分析
+ 服务端的
  - 动作频率：角色执行特定行为动作的数量，例如FPS游戏中角色攻击目标的次数
  - 运动轨迹：角色在游戏中的移动轨迹
  - 社交活动：角色在游戏中的社交活动，交易等。




#### 论文
1. Ahmad, M.A., Keegan, B., Srivastava, J., Williams, D., Contractor, N.: Mining for gold farmers: automatic detection of deviant players in MMOGs.
2. Bernardi, M.L., Cimitile, M., Martinelli, F., Mercaldo, F.: A time series classification approach to game bot detection
3. Kang, A.R., Jeong, S.H., Mohaisen, A., Kim, H.K.: Multimodal game bot detection using user behavioral characteristics
4. Prasetya, K., Wu, Z.D.: Artificial neural network for bot detection system in MMOGs.
