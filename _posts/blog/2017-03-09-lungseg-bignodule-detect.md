---
layout: post
title: 肺部CT大结节检测
description: 图像处理
category: blog
---

## 0 摘要

**目的**：当前使用CAD系统检测对肺结节检测时，只在较小结节上有很好性能，通常不能检测更少的更大的结节的，这些可能是癌性。我们专门为大于10mm的solid结节检测设计了算法。

**方法**：我们提出的检测流程，初始时由三维肺分割算法，通过形态学操作包含入黏着在肺壁上的大结节。额外的处理是mask out肺空间的外部结构，以确保肺和实质结节有着相似的轮廓。接着，通过多阶段的阈值和形态学操作获得结节候选，以检测大结节和小结节。对每个候选分割之后，计算出一个基于强度、形状、blobness、和空间结构的24个特征的集合。使用一个径向偏置SVM分类器对结节候选分类，在完全公开的肺部图像数据集上使用10折交叉验证评估性能。

**结果**：本文提出的CAD系统弄获得了98.3%(234/238)的灵敏度，94.1%(224/238)大结节的准确率，平均4.0和1.0 FPs/scan。

## 一  数据集

本论文中，使用的是LIDC-IDRI数据集。数据集包含了来自7个机构的1018个病例异构数据。在第一轮盲读阶段，每个可疑病变被标注同时被分类为 $non-nodule,nodeul<3mm,nodule\ge 3mm$。对于$nodules\ge 3mm$的被充分表征为3D分割，会提供其半径和形态学特征描述。

我们使用区域厚度小于等于2.5mm的，比这更厚的区域数据被丢弃，因为我们将这些数据定义为质量不足。尽管slice厚度为3mm的大结节依然可以被检测到，最近的临床指南推荐使用thin-slice,因而我们也使用thin-slice CT scan。除此之外，包含不连通的slice空间的也被丢弃，最后获得了888个适于分析的CT scan。

下表是数据概况

|Aggrement levels|Nodules $\ge 3mm$|Nodules $\ge 10mm$|solid nodules $\ge 10mm$|
|---|---|---|---|
|At least 1|2290|393|322|
|At least 2|1602|325|277|
|At least 3|1186|269|238|
|At least 4|777|199|172|

从888个scan中制作了36378个标注。由于结节可能被多个读者标注，不同的读者的标注在近似位置（小于标注的半径之和）的会被合并。只对 $nodule\gt 3mm$的结节标注做合并，因为这个作为引用标准。其半径、容积、和坐标都平均化。我们仅仅选择被分类为潜在恶性结节的作为大结节的引用标准。根据Dutch-Belgian NELSON肺癌实验，潜在恶性结节指的是容积大于$500mm^3$的，其对应的半径为近似 10mm。然后，选择被大多数读者(4个中的3个)接受的结节，获得了269个结节。并且设计CAD系统只关注 solid 结节的。由读者打分的各种各样的形态学特征用来定义结节类型。本文中，只有当大多数读者给出的上下文特征分数高于3(1=ground-glass/nonsolid,3=part-solid,5=solid)结节被认为是solid。最终获得了238个solid结节，这形成了分析所用的最终结节集合。

评估过程中，CAD标记的位置在标注半径范围内的可被认为是命中了。如果CAD标记命中了引用集中的结节，则被分类为正样本(True  Positive)。不在病变引用集上的标记(即$nodule\ge 10mm$只被一两个放射医师接受，$nodules\gt 10mm$,subsolid nodules,non-nodules)被认为是不相关的，且没有被记入为假阳性(False Positive).

## 三 方法

算法的主要流程如下，算法的开始时肺分割。肺分割被细化以将那些与肺壁相连的结节包含进来。

![](/images/blog/lung_bignodule1.png)

 之后的后续处理步骤是移除肺外部的背景并重新将图像抽样为各向同性分辨率。候选检测阶段确认大结节候选的位置并为每个候选构建segmentation。使用特征抽取来获得候选的判别特征，这在分类阶段将被用来将候选分为结节和非结节。

### 3.1 肺分割

CAD系统的第一步的肺分割用来决定肺的ROI(region of interest)。主流的算法依赖于一种基于阈值的方法。由于相似的强度特征，粘附在肺壁的结节通常无法包含入肺分割内。

我们使用Rikxoort提出的算法作为初始的肺分割segmentation。方法由这些组成：大气道抽取、肺分割、左右肺分割和segmentation 平滑。当肺分割算法失败时，我们通过手动衡量输入参数，比如气管的seed point来更正肺分割。

为了在肺分割segmentation包含入大的肺结节，使用了额外的细化的肺分割segmentation。肺结节被剔除的区域通常看起来像在肺segmentation的表面上有个洞。我们实验了两种方法**(1)**使用形态学逻辑滚球rolling-ball操作**(2)**在rolling-ball操作之后做扩张操作。所有的形态学操作使用的是球形结构元素。结构元素的半径 $d_{struct}$设置为一个x维度在输入scan中的比率。我们评估了 $d_{struct}={2%,3%,4%,6%,8%}$ 的rolling-ball操作和 $d_{struct}={1%,2%}$ 的扩张操作。

### 3.2 预处理 

肺segmentation用作mask，segmentation之外的区域被设置为肺组织的平均强度(-900HU)。这可以避免肺病变看起来与其内部的结节非常不同，这就要求有不同的或者额外的特征来准确检测这些结节。肺segmentation使用高斯过滤器重新取样为1.0mm的    各向同性分辨率(isotropic resolution)。

### 3.3 候选检测

候选检测步骤旨在局部化所有肺内部的结节。这个任务之所以艰难，是因为结节在形态上、尺寸上和强度上变化范围太广。本文中，候选检测由三部分组成:初始候选检测，连通组件分析，结节检测细化。

#### 3.3.1 初始候选检测

与其他相似度密度较高的结构(主要是血管)相比，大结节通常有着十分不同的形态学特征，一个简单的阈值和形态学开运算就足够充分检测大部分大结节。然后，由于它们的尺寸，与其他非密度结构（大部分是胸膜壁和脉管系统）相比大结节倾向于内部相连。这使得检测器的参数选取变得复杂，尤其是形态学开运算的结构元素的尺寸。相连结构的尺寸可能变化范围广。举一个例子，一个大的结构元素需要移除结节上的大结构的黏着(attachment of large structure)，但是会导致小结节无法检测。

为检测不同尺寸的候选，使用多阶段阈值和形态开运算。强度阈值为-300HU来区分solid nodule。每个阶段，顺序使用半径为9,7,5或者3mm的球形结构的扩张操作，并产生了中间候选mask。候选检测始于大结构半径并渐渐使用更小的半径。为防止先前候选与新候选合并，在处理下一阶段之前先前候选使用了一个3mm的保障边界。在阶段n计算候选检测，新的中间mask与阶段n-1的输出mask合并，使用逻辑与操作。下图示意了候选检测的输出。每一阶段使用不同的阈值和形态扩张操作，mask的结果与上一阶段的mask合并。第一行是检测大结节，第二行是小一点的结节检测过程。

![](/images/blog/lung_bignodule2.png)
 
#### 3.3.2 连通组件分析

使用初始候选检测之后，所有相连体素使用连通组件分析聚合为候选。为了移除cluster size超出目标范围的，我们丢弃了cluster容积小于 $268mm^3$的和大于 $33524mm^3$的，其分别对应的完美球形半径为8mm和40mm。

#### 3.3.3 结节segmentation

初始候选指示了候选cluster的坐标。作为候选的一个准确的容积和形态学评估，量化结节特性尤为重要，我们使用了由Kuhnigk提出的robust结节分割方法。给定cluster上的立方块输入容积和seed point，算法进行区域增长并使用有效的形态学开运算来从血脉组织和胸腔壁分离结节。对每个候选cluster，从cluster的主轴(major axis)上获得seed point。VOI(volume of interest)被定义为初始边长为60mm的在cluster附近的立方块。立方块的尺寸是自动适应的，必要的话，推荐更大的结节。

为进一步移除大于或小于预定目标的候选，我们使用3.3.2节中相同的阈值。为避免结节中出现重复结节，与任何已经被接受的segmentation位置小于10mm的都被丢弃。结节segmentation的结果集合将进一步用作特征抽取和分类。

### 3.4 特征抽取

特征抽取用来获得可以区分结节和非结节的特征。定义了四个不同集合的特征：强度、cluster、blobness、和上下文特征。下表展示了所有特征的概览

**强度特征**

|ID|强度特征|Notes|
|---|---|---|
|1-3|Density inside candidate segmentation (mean,stdev, entropy)||
|4-6|Density inside bounding box (mean, stdev, entropy)||
|7-9|Density around candidate segmentation (mean,stdev, entropy)|within 8mm outside segmentation|

**cluster feature**

|ID|cluster feature|Note|
|---|---|---|
|10|Volume of candidate segmentation $V_{cand}$||
|11|Diameter of candidate segmentation $D_{cand}$|The longest diameter on axial plane|
|12|Sphericity: ratio of candidate’s volume inside sphere S to the volume of sphere S |Sphere S is centered at the candidate location with diameter $D_{cand}$|
|13|Compactness1: $V_{cand}/(dim_x ·dim_y ·dim_z)$ | $dim_i$ is the width of bounding box indimension i |
|14|Compactness2: $V_{cand}/((max(dim_x, dim_y, dim_z))^3$)||

**Blobness feature**

|ID|Blobness feature|Note|
|---|---|---|
|15|Maximum flter response|Features are computed using the flter|
|16-17|Filter response inside candidate segmentation(mean, stdev)|response of Li blobness flter(scale: 2, 5, and 8 mm)|
|18-19|Filter response inside bounding box (mean, stdev)||

**Context feature**

|ID|Context feature|
|---|---|
|20|Distance to pleural wall|
|21-23|Distance to carina in X, Y, and Z|
|24|Distance to top of the lung in Z|

#### 3.4.1 强度特征

强度特征用来量化候选segmentation内外区域的强度特征。特征集直接计算自各向同性（isotropically）重新取样的CT scan。使用了三个不同的区域**(1)**候选segmentation内的区域**(2)**候选segmentation的bounding box内的区域**(3)**在候选segmentation周围距离小于8mm的区域。每个区域，计算均值、标准差和体素强度熵。

#### 3.4.2 cluster特征

cluster 特征从候选segmentation中计算得来。这些特征集由半径、容积、球形性、紧凑性1(compactness1)和紧凑性2(compactness2)。

半径是通过segmentation的轴状位的部分最长的轴得到。容积通过计算cluster size的$mm^3$。为计算球形性，与候选有着相同容积的球S在候选质心处构建。球形性定义为候选容积在球S内与球形S的容积的比例。紧凑性为候选容积与候选segmentation周围的bounding box容积的比例。使用了两个不同的boundiing box。紧凑性1使用的bounding box为在全部x,y,z维度包容候选segmentation的最小box。紧凑性2使用的bounding box定义为立方块，其尺寸为候选segmentation的最大维度的大小。

#### 3.4.3 blobness特征

blobness特征广泛用于增强结节结构和提高结节检测的灵敏度。我们使用的是由**Li,Sone 和 Doi**开发出来的结节增强过滤器，并偶从过滤图中抽取blobness特征。

算法通过对输入图像进行高斯kernel的二阶导数做卷积来计算Hessian矩阵。给定Hessian矩阵，三个特征值，定义为 $\lambda_1=f_{xx},\lambda_2=f_{y},\lambda_3 = f_{xy}$，其中 $\lambda _1>\lambda _2>\lambda _3$。增强过滤器的最终输出是通过计算 $Z(\lambda _1,\lambda _2,\lambda_3)=\|\lambda _3\|^3/\|\lambda _1\|\quad if\quad \lambda _1<0,\lambda _2<0,\lambda _3<0$；否则为0。为了获得范围分布广泛的结节尺寸，使用来了多尺寸增强过滤器。我们对高斯kernel定义了三种(2mm,5mm,8mm)，这是基于要增强的目标结节的经验值。结果会有三种不同的输出图像，是一种从所有图像中选择最大值的结合。

blobness特征的抽取通过衡量**(1)**过滤器的最大输出**(2)**结节segmentation内部的输出的均值和标准差**(3)**bounding box内部输出的均值和标准差。

#### 3.4.4 上下文特征

上下文特征描述了候选与其他肺结构如胸膜壁，隆突和肺上部的相对位置。根据它们的位置，肺结节和假阳性候选可能有着不同的形态学特征和恶性概率。结节粘附于更加刚性的结构上时，结节更可能是细长的。**Horeweg**的研究表明，大部分病变结节位于costal-hilar半径的外部的三分之一的上叶（at outer one-third of the costal-hilar diameter and at upper lobes ）。意味着这些区域应该引起注意。到carina的距离和肺的顶部区域需要给予更多的权重。

为计算相对于胸膜壁的距离，我们做了在肺分割内做了距离变换并抽取候选质心中心的值。carina点的检测方法是，找到轴向(axial)区域中气管被一分为二的地方，在分叉处周围选取气管区域，并抽取其质心。从carina到坐标轴的相对X，Y，Z距离作为特征。从候选到肺顶部的相对距离的计算方法为，计算候选的Z轴世界坐标到肺分割的top slice的Z轴世界坐标的距离。

#### 3.4.5 分类器

对特征向量的分类器，使用了径向偏置核函数SVM-RBF。其中的C和伽马参数在训练集上的网格搜索的10折交叉验证得到的。C定义了正则化参数，伽马定义了RBF核函数的宽度。搜索区间为 $C={2^1,2^2,2^3,...2^12}$，并且$\Gamma ={2^{-12},2^{-11},...2^{-5}}$。本文使用的是LIBSVM实现。分类之前先对所有特征进行正则化，均值为0，单位标准差。

## 四 结果

评估CAD系统检测大结节的性能。在病人级别的10折交叉验证，分类器在嵌套的交叉验证中优化得到。

### 4.1 肺分割

888个scan中有12个分割失败，因为气管处的seed point没有被正确检测到。这可以通过手动提供seed point来修正。下表显示了在CAD系统上使用细化的肺分割之后的影响。肺结节的位置坐标如果在segmentation内部，则其分类为包含在肺segmentation内。算法的评估方法为，被检测到的结节的数量和候选的数量。

**Without additional lung segmentation refinement**

|Rolling ball(kernel size (% of image))|Dilation(kernel size (% of image))|Inside segmentation (%)|After candidate detection(%)|Candidates/Scan|
|---|---|---|---|---|
|-|-|84.9|87.8|39.5|

**With additonal lung segmentation refinement**

|Rolling ball(kernel size (% of image))|Dilation(kernel size (% of image))|Inside segmentation (%)|After candidate detection(%)|Candidates/Scan|
|---|---|---|---|---|
|2|-|84.9|87.8|39.5|
|4|-|97.1|98.3|47.4|
|6|-|98.3|99.2|56.6|
|8|-|99.6|98.7|63.2|
|2|1|88.7|95.4|120.7|
|4|1|99.6|97.9|133.1|
|6|1|100.0|96.6|145.5|

上表显示没有使用额外的细化算法时，肺分割只有84.9%的大结节被包含入。使用额外的细化算法时可以减少被排除的结节数直至所有的结节都被包含入内。但是也可以看到，细化算法会增大肺segmentation，它可能会引入不相关区域(比如肺壁,hilar)，这会恶化候选检测的性能。因而，我们同时评估了候选检测的灵敏度。数据集中至少3个放射医师标注的，包含全部 $nodules\ge 10mm$的287个scan的数据集。实验表明，初始肺分割之后，使用 $d_{struct}=6%$ rolling-ball操作和 $d_{struct}=0%$的扩张操作，可以使得候选检测获得最高的灵敏度和差不多数目的 candidates/scan。此配置应用于后续的实验中。

### 4.2 候选检测

888个 CTscan中，候选检测平均生成48.3 candidates/scan，包括了99.2%（236/238）的全部大结节。此集合用作进一步的分类 任务。不同准入水平的候选检测性能如下表。结节的候选检测的灵敏度随着放射医师的准入水平的上升而增加。

|Aggrement levels|Solid nodules >10mm|Detected nodules (%)|Candidates/scan|
|---|---|---|---|
|At least 1|322|97.2|48.3|
|At least 2|277|98.9|48.3|
|At least 3|238|99.2|48.3|
|At least 4|182|100.0|48.3|

### 4.3 特征抽取和分类

CAD系统在包含大结节数据集上的不同准入水平的FROC曲线如下图左图所示，右图表示的 是CAD系统包含或不包含不相关的发现(被当做false positive假阳性的)时的性能。假阳性的数目现实的是求对数之后的结果。

![](/images/blog/lung_bignodule3.png)

为了量化比较，不同假阳性率的平均灵敏度如下表。CAD系统识别分别在 1FPs/scan和 4FPs/scan时识别了94.1%(224/238)和98.3%(234/238)的大结节。注意到分类阶段的最大灵敏度收敛到候选检测阶段的灵敏度，为99.2%(236/238)。这意味着与检测到的候选相关的，分类阶段的准确分类率在 1FP/scan和4FPs/scan分别为 94.9%和99.1%。

|Agreement levels| 1/8 |1/4 |1/2| 1| 2| 4| 8| Average|
|---|---|---|---|---|---|---|---|---|
|At least 1 |0.773| 0.804| 0.842| 0.879| 0.913 |0.950| 0.960 |0.874|
|At least 2 |0.841 |0.866 |0.895 |0.924 |0.949 |0.978 |0.982 |0.920|
|At least 3 |0.874 |0.895 |0.916 |0.941 |0.962 |0.983| 0.992 |0.938|
|At least 4 |0.929 |0.940 |0.956 |0.978 |0.995 |1.000 |1.000 |0.971 |


### 4.4 与已有的CAD系统对比

ISICAD系统获得了ANODE09的最高排名，用作对比。两个CAD系统的FROC曲线如下图，候选检测阶段检测到99.2%(236/238)(本文算法)和84.9%(202/238)(ISICAD)。在 1FP/scan时，63.0%(150/238)更多的大结节被本文提出的算法正确分类。不过要注意的是两个CAD系统为不同类型的结节设计的。

![](/images/blog/lung_bignodule4.png)
  














