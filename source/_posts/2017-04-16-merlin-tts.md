---
layout: post
title: 使用merlin从头构建你的声音
description: 语音合成
category: 语音
---

本文参考自: http://www.speech.zone/exercises/build-a-unit-selection-voice/label-the-speech/
而该文章又参考自爱丁堡大学的一篇论文的思路，[论文](http://www.cstr.ed.ac.uk/downloads/publications/2004/clarkrichmondking_ssw504.pdf),[论文实现工程文件](http://www.cstr.ed.ac.uk/downloads/festival/multisyn_build/) 。下面的教程中直接使用了很多此工程中的文件，如果在教程中没找到对应的文件夹，可能需要下载此工程。
## 一  需要的工具

+ python:2.7

+ Festival:一个语音合成工具 [Festival](http://www.cstr.ed.ac.uk/projects/festival/) 

+ Edinburgh Speech Tools: 一些地方也称为Festival，下载之后为speech_tools/ 文件夹

+ 其他依赖：详见Festival或Edinburgh Speech tools的说明文档。以及`lib32ncurses5-dev`和`libX11-dev`(linux安装)

+ HTK :语音识别工具，如果安装是3.4，需要修复这个 [bug](https://github.com/JoFrhwld/FAVE/wiki/HTK-3.4.1)

##  二  介绍

本文主要关注于合成流程中的波形生成器阶段，尽管可以在前端工具(festival和HTK)做对应的修改，即在发音词典中加入新的词。
**不要在windows上玩这个**

### 2.1 本文主要流程:

1. 选取recording脚本
2. 在studio中做recording
3. 准备好workspace
4. 将录音转换为要求的格式，并仔细检查
5. label 语音文件
6. Pitchmark 语音
7.创建声音
8.评估声音文件

### 2.2 相关文章

文字转语音的流程架构,[TTS流程](http://www.speech.zone/pipeline-architecture-for-text-to-speech/)

[speech论坛](http://www.speech.zone/forums/forum/speech-synthesis/)

## 三 准备好workspace

假设所有的文件将放在`/workspace/merlin`文件夹下，下载并解压此文件[SS](http://www.speech.zone/wp-content/uploads/2015/12/ss.zip),文件组织结构如下:

```
xiatao@sjkxbgpu:~/workspace/merlin$ cd ss
xiatao@sjkxbgpu:~/workspace/merlin/ss$ tree ./
./
├── pm
├── recordings
├── setup.sh
├── utts.data
├── utts.pauses
└── wav

3 directories, 3 files

```

如果所需工具都已经正确安装，现在需要编辑`setup.sh`文件中的两个变量`SSROOTDIR`和`FROOTDIR`，分别指向安装工具目录和festival安装目录。原文件的两个变量值为:

```
SSROOTDIR=/Volumes/Network/courses/ss/
FROOTDIR=/Volumes/Network/courses/ss/festival/festival_mac
```
修改为:

```
SSROOTDIR=/home/xiatao/my_soft_install/merlin/tools
FROOTDIR=/home/xiatao//my_soft_install/merlin/tools
```
我对应的目录组织结构如下:

```
xiatao@sjkxbgpu:~/my_soft_install/merlin/tools/festival$ ls
ACKNOWLEDGMENTS  bin  config  config.cache  config.guess  config.log  config.status  config.sub  configure  configure.in  COPYING  doc  examples  INSTALL  install-sh  lib  Makefile  make.include  missing  mkinstalldirs  NEWS  README  src  testsuite
xiatao@sjkxbgpu:~/my_soft_install/merlin/tools/festival$ cd ..
xiatao@sjkxbgpu:~/my_soft_install/merlin/tools$ ls
bin  compile_tools.sh  festival  festvox  INSTALL  install_package_and_shell  readme  speech_tools  SPTK-3.9  WORLD  WORLD_v2
```
然后执行:```source setup.sh```。此命令不会有任何输出，但是相关变量会被配置。本文流程走完之后，会有如下目录以及其对应的内容:

|Folder |   Contains|
|---|---|
|recordings |   speech recordings, copied from the studio|
|wav |   individual wav files for each utterance|
|pm   | pitch marks|
|mfcc |   MFCCs for use in automatic alignment|
|lab  |  label files from automatic alignment|
|utt   | Festival utterance structures|
|f0   | Pitch contours|
|coef |   MFCCs + f0, for the join cost|
|coef2  |  coef2, but stripped of unnecessary frames to save space, for the join cost|
|lpc  |  LPCs and residuals, for waveform generation|

## 四 recording脚本

鉴于单元(音素)选取极端依赖于数据库内容，我们需要仔细考虑应该做哪些recording。

我们需要选取一个recording脚本。标准方法是一句句的贪心选取句子，从一个大的文本语料库（比如，小说或报纸）来尽可能覆盖最多的语音(以及可能的韵律)。本文不会直接走这一步骤，而是直接使用已经存在的[CMU ARCTIC](http://festvox.org/cmu_arctic/)

如果记录全部的内容（CMU语音所读的文本），大概会得到一个小时的语音。但是建议先从记录(读)A集合的593个prompts 开始，并创建对应的语音

### 4.1 关于CMU ARCTIC数据库

数据库包含了从静心挑选的文本里录的1150个语音，包含了US English男性(bdl)和女性(slt)播音员。

`cmuarctic.data`包含了1132个句子-语音准备清单，其内容示例如下:

```
( arctic_a0001 "Author of the danger trail, Philip Steels, etc." )
( arctic_a0002 "Not at this particular case, Tom, apologized Whittemore." )
( arctic_a0003 "For the twentieth time that evening the two men shook hands." )
( arctic_a0004 "Lord, but I'm glad to see you again, Phil." )
...
```

它既包含了16khz的波形(waveform)语音，同时也包含了EGG信号。全部的labeling由基于`Festvox`(festival依赖的一个工具)的labeling脚本完成。数据库中已经包含了完整的可运行的Festival Voices.

**CMU ARCTIC Database**

+ US Enlish bdl(male)(0.95)
+ US English slt(female)(0.95)
+ US English clb(female)(0.95)
+ US English rms(male)(0.95)

### 4.2 使用说明

建议下载上面的`ARCTIC`数据库中的`slt`，它们是完整的Festival Voices，所以需要删除其他的而只保留波形文件（wav）和`utts.data`。将这些文件(wav和utts.data)复制到workspace目录下。

#### 4.2.1 utts.data文件

utts.data文件是Festival用来定义unit(音素)选取的数据库。它的内容示例如下:

```
( arctic_a0001 "Author of the danger trail, Philip Steels, etc." )
( arctic_a0002 "Not at this particular case, Tom, apologized Whittemore." )
( arctic_a0003 "For the twentieth time that evening the two men shook hands." )
```

其中的每一行代表 了数据库中的一个utterance(语音)。格式如下:

1.一个开的插入语,`(`
2.一个唯一的识别码，被用作任何与此utterance（音素）相关的文件的文件名。ARTCTIC的文件名形式为 `arctic_lnnnn`，我们自己准备的材料可以以`yourname_nnnn`这种形式，其中`nnnn`以0001开始。
3. 文本，双引号内部
4.关闭的插入语`)`

#### 4.2.2 添加自己的素材

由于ARCTIC脚本给的是通用的双音，合成全部类型的句子并不完美。你可以通过添加特定领域的自己的声音来提高它的自然性，添加更多的素材到数据库。这一步并不是一定要做的。

建议添加一些prompts 以包含如下：

1. 在不同上下文环境里包含了自己名字的5个句子（比如，句子的开始和结尾处）
2.10个短的高频使用的短语集，比如"Hello""Hi""How are you"等等。
3. 大概50个可以覆盖某个很小领域的句子

选择一个可以获取词汇表中所有单词的有限领域，其中每一项（单词）多次发音（不同的上下文和位置），并且在50句话内覆盖。根据这些，你可以合成更广泛的新句子。示例:

+ 时间和日期

+ 街道地址，街道名很少。

设计完自己的领域素材之后，准备一份新的与utts.data相同的文件，包含了自己的句子。此文件将用于SpeechRecorder tool来记录这些句子。不必包含ARCTIC 脚本：它已经被包含入SpeechRecorder。除此之外，需要在utts.data的末尾添加自己的新句子。

**注意：使用一个文本编辑器来创建这些文件，避免出现非ASCII字符**

## 4.2.3 自动文本选取

这是一个可选的步骤，用以实现自己的贪心文本选取算法

与从限定领域手工添加小集合的句子相比，我们可以实现一个简单的文本选取算法来选取其他素材。

此步骤可以让我们选取与ARCTIC的A集合同等大小（记录的语音）的额外数据集来形成自己的素材。

切记：仍然需要在studio中记录这些素材。

你可以：

+  抽取与ARCTIC文本选取算法完全一样的副本，或者自己实现

+ 可以使用更新的文本数据源，而不是ARCTIC里的古老小说

+ 如果你选取特定领域的源文本库，就可以创建一个较好的特定领域声音。其中的难点在于找到足够大的文本（比如，爬虫）


## 5  make the recording

即在录音棚中读文本，并录音。

## 6 准备录音

16khz的波形文件即可。

SpeechRecorder tool在 文件名前加 了前缀。需要移除，使得文件名完全匹配utts.data中的utterance 识别码（前缀无非是"_1","_2"等）。不要重复记录发音utterance 

**下采样**
下面的是下采样单个文件的示例，保存为所需的RIFF格式(Linux为wav格式)

```
bash$ ch_wave -otype riff -F 16000 -o wav/arctic_a0001.wav recordings/arctic_a0001.wav
```
你需要些个shell脚本来处理所有的在`recording`目录的文件。如果你的record数据是24bit而非16bit，则需要使用`sox`来改变bit深度，并同时下采样：

```
bash$ sox recordings/arctic_a0001.wav -b16 -r 16k wav/arctic_a0001.wav
```

## 7 标记语音

使用text-to-speech（TTS） 系统的前端(front-end)工具从文本中来获取标签，接下来需要将它们 与已经记录的语音对齐，此处使用了自动语音识别处搬来一个技术。

在继续之前，需要确保已经完成如下步骤：

1. 至少完成了recording ARCTIC中的"A"集合
2. utts.data文件。
3. 包含了wav文件的目录，wav文件都在utts.data中有对应。
4. 确保文件名和数字是正确的。

下一步是为语音创建时间对齐语音标签，使用强制对齐和HTK语音识别工具。首先得为HTK设置一个目录结构(安装HTK时会有个HTKDemo的目录，按照其目录创建对应的目录)，运行:

```
bash$ setup_alignment
```

这一步骤会创建一个`alignment`目录包含了HTK相关的目录。脚本会告诉你需要创建其他文件，这在下一步。

### 7.1 选择词典和语音集

选一种口音的英语。此处的选择将会决定对齐的余下部分所使用的词典和语音集。此处我们假设使用的是British English词典**unilex-rpx**，如果使用的是不同的词典，你需要在所有命令中替换"unilex-rpx"为下面的其他选项:

+ unilex-gam – General American English

+ unilex-rpx – British English (RP)

+ unilex-edi – Scottish English (Edinburgh)

**定义语音集**

将定义了语音集的文件复制到对齐(alignment)目录。注意变量`$MBDIR`来自`setup.sh`文件内定义，执行完`source setup.sh`即可。

```
bash$ cp $MBDIR/resources/phone_list.unilex-rpx alignment/phone_list
bash$ cp $MBDIR/resources/phone_substitutions.unilex-rpx alignment/phone_substitutions
```
phone_list包含的是语音集的列表。包含了一些特殊的语音，这在自动语音识别中很常见。如果X是一个stop或者破擦音，那么X_cl被加进来来标记局部闭合。标签`sp`（short pause）加进来作为词中间的静音，`sil`代表更长时间的静音（每个utterance(语音)的开始和结束）。

phone_substitutions文件包含了aligner允许的可能的替换列表。这些被限定为元音reduction（减少），比如规则`aa@`代表"aa"可以被标记为`@`(schwa),如果基于声学模型它是一个更可能的标签。

**处理不在词典中的单词**

鉴于强制对齐会从语音中产生语音标签以及它们的单词抄录(transcriptions),它需要知道每个单词的抄录。在语音合成中可以在所有未知单词中使用letter-to-sound规则，但是对于标记语音数据不精确。切记在语音记录数据库的任何错误会对语音合成产生直接的影响。

**对着词典检查脚本**

```
bash$ festival $MBDIR/scm/build_unitsel.scm
festival>(check_script "utts.data" 'unilex-rpx)
```

festival会告诉你哪些不在词典里的单词，以及按照letter-to-sound的规则它该如何发音。找到不在词典中的单词之后，创建一个文件my_lexicon.scm，格式如下(注意第一行中的词典名称，不同的词典名称不一样):

```
(setup_phoneset_and_lexicon 'unilex-rpx)
 
(lex.add.entry '("pimpleknuckle" nn (((p i m) 1) ((p l!) 0) ((n uh) 1) ((k l!) 0))))
(lex.add.entry '("womble" nn (((w aa m) 1) ((b l!) 0))))
```

为获得正确的单词发音，启动festival,并执行check_scipt脚本命令（参考上面），以确保正确的词典被载入。然后使用命令 `lex.lookup`找到近似发音单词来构建你的语音。如果有很强的非本地口音，别尝试匹配你所使用的真实声音，而尝试写与其他类似单词的发音一致的发音。如果此阶段没有添加任何发音，创建一个空白文档即可。

```
bash$ touch my_lexicon.scm
```

### 7.2 时间对齐标签

数据库需要时间对齐标签。保持标签与前端在运行时做出的预测的一致性很重要， 因而我们需要使用相同的前端来创建初始标签序列，然后使用强制对齐在这些标签中加入时间戳。

初始的用于强制对齐的语音序列来自Festival,通过运行前端脚本。如果使用不同的词典，注意修改`unilex-rpx`。

**创建初始标签**

```
bash$ festival $MBDIR/scm/build_unitsel.scm ./my_lexicon.scm
festival>(make_initial_phone_labs "utts.data" "utts.mlf" 'unilex-rpx)
```

输出文件utts.mlf，是一个包含了的utterances(语音)的语音转录(transcription)的HTK master label file(MLF)；其标签暂未与波形(waveform)时间对齐。

如果想要设计自己的 脚本，以上命令是最简单的方式来将文本转换为语音序列，这样就可以衡量覆盖率。

强制对齐涉及到训练 HMMs，正如自动语音识别。因此，语音需要参数化。我们所使用的特征是MFCCs。

**抽取MFCC**

```
bash$ make_mfccs alignment wav/*.wav
```

**对齐**

```
bash$ cd alignment
bash$ make_mfcc_list ../mfcc ../utts.data train.scp
bash$ do_alignment .
```

(注意最后一个命令的空格和点号)

do_aligner命令将会持续超过20分钟，取决于机器配置和记录的语音。对齐结束后，需要产生的MLF文件切分为Festival能使用的独立的标签文件，此时它已经为标签包含了正确的时间对齐。

**切分MLF文件**

```
bash$ cd ..
bash$ mkdir lab
bash$ break_mlf alignment/aligned.3.mlf lab
```

### 7.2.1 修改对齐脚本

此步骤不是必须，最好是觉得合成结果比较差时回头参考此步骤来调整。

修改do_alignment脚本会影响强制对齐的质量。修改脚本之前，首先得找到脚本位置，并复制一份。```which do_alignment``` 。编辑此文件之后再运行。

**在数据子集上训练，但在整个数据集上对齐**

你需要创建一个train.scp文件，此文件中只包含了需要在模型上训练的MFCC文件列表。假设该文件名为train_subset.scp，执行如下命令:

```
HERest -C config -T 1023 -t 250.0 150.0 1000.0 -H hmm${i}/MMF -H hmm0/vFloors -I aligned.0.mlf -M hmm$[$i +1] -S train_subset.scp phone_list
```

**改变混合组件的数量**

默认模型的输出概率密度分布是8个组件的混合。HTK使用一个称为`mixing up`的方法逐步增加组件数量，此处我们从1到2，然后是3,5，最后到8个组件。可以修改脚本中的此行:

```
# Increase mixtures.
 
for m in 2 3 5 8 ; do
```

**改变元音reductions**

不必修改do_alignment就可以达到，只需要修改phone_substitutions文件。尝试移除所有的"substitutions"（比如，创建phone_substitutions和空白文件）


## 8 Pitchmark语音

用于波形连接的信号处理是音调(pitch)同步的，因此语音数据库必须包含独立的语调周期标记。`make_pm_wave`可以用来从波形语音生成音调标记，其中的`-[mf]`代表选择其一`-m`(男性)或`-f`(女性)。

```
bash$ make_pm_wave -[mf] pm wav/*.wav
bash$ make_pm_fix pm/*.pm
```

`make_pm_fix`用于调整语调标记，使得他们在波形中对齐于一个peak极点，并且插值在语音的无声区域。

**查看音调标记**

为查看音调标记（精确检查），需要将他们转化为标签文件。这些可以在与波形对应的波浪中看到。

```
bash$ mkdir pm_lab
bash$ make_pmlab_pm pm/*.pm
```

**调整音调标签设置**

在自己的声音上需要调整一些参数。此时需要复制脚本 `$MBDIR/bin/make_pm_wave`到当前`/ss`目录下，并对该文件做修改。记得对应的运行命令是，运行当前的`make_pm_wave`。

找到脚本中的`DEFAULT_PM_ARGS `，其中的`min`和`max`是音调标签之间的最小和最大值(例如，音调周期(period)是1/F0)。`1x_1f`和`1x_hf`的是值单位为Hertz的频率以及一个控制过滤器，该过滤器移除在音调标签之前的高频和低频。

如果决定修改默认值，需要找到你的声音的`F0`值域。可以使用`Praat`的编辑命令来检验一些音调轮廓的波形，并记录最大和最小值（剔除由Praat产生的明显纰漏）。如一段中设置高频和低频过滤器掉不在此音调范围内的值，同时为你的声音设置合适的max和min值。再次运行音调标记器并检查输出。需要将pitchmark转换为标签文件来查阅。

在调整pitchmarker设置时， 删除`-fill`选项。

## 8  合成声音

### 8.1 Utterance(发声)结构

Festival中的目标代价函数使用语言学信息来计算，所以我们需要提供语音数据库中所有候选单元的信息，这些信息存储在发声结构中。发声结构包括语音字符串，将这些语音和它们的parent（双亲）音节和单词连接的树形结构，等等。我们将由强制对齐获得的语音时间戳加入到这些结构中。

首先，创建发声结构:

```
bash$ mkdir utt
bash$ festival $MBDIR/scm/build_unitsel.scm my_lexicon.scm
festival>(build_utts "utts.data" 'unilex-rpx)
```

然后，运行和分析语音持续时间分布，并标记任何异常值。

```
bash$ mkdir dur
bash$ phone_lengths dur lab/*.lab
bash$ festival $MBDIR/scm/build_unitsel.scm
festival>(add_duration_info_utts "utts.data" "dur/durations")
```

### 8.2 音调(pitch)追踪

join cost(目标损失函数)的一个重要组件是基准频率，`F0`。这个从音调标记中独立抽取的，尽管二者很显然是紧密相关的。

而音调标记是波形生成所使用的信号处理必备的，音调轮廓(更准确的说是，`F0`轮廓)是join cost必须的。

```
bash$ mkdir f0
bash$ make_f0 -[mf] wav/*.wav
```

其中`-[mf]`,`-f`指女性，`-m`指男性。

### 8.3 join cost系数

join cost衡量的是数据库中的候选单元join处的潜在声音不匹配。为了在合成声音运行时更快，可以预处理用于计算join cost的声学特征。

Festival的join cost衡量的是spectrum(范围)(即MFCCs)的不匹配和F0的不匹配。接下来是对每个发声utterance规范化并结合MFCCs和F0为单个文件。

```
bash$ mkdir coef
bash$ make_norm_join_cost_coefs coef f0 mfcc '.*.mfcc'
```

并且由于join cost仅使用每个候选单元的第一和最后一帧来评估，这些文件可以被剥离掉所有不在双音边界附近的值了，这使得文件变得更小、更快地载入到Festival。

```
bash$ mkdir coef2
bash$ strip_join_cost_coefs coef coef2 utt/*.utt
```

### 8.4 波形表征

尽管单元选取对于预先记录的波形片段的拼接至关重要，我们仍然可以为源过滤模型参数存储这些波形文件。

Festival在语音合成时所使用的语音表征是残差激励线性预测系数(RELP)。这样就可以操作spectrum(范围)和F0（比如，在拼接处）以及持续时间。然而，实际上Festival的Multisyn引擎没有做任何操作。

```
bash$ mkdir lpc
bash$ make_lpc wav/*.wav
```

## 9 运行声音

为了运行声音，启动Festival并载入声音(将的rpx修改为合适的名称)

```
bash$ festival
festival>(voice_localdir_multisyn-rpx)
festival>(SayText "Hello world.")
```

## 10 提高

前面的步骤已经完成声音的构建，接下来是如何提升声音质量了，我们所使用的办法是合成多版本的声音，然后对比和测试。

构建多版本声音的方法就是完全复制一份`ss`文件夹，同时对`wav`,`mfcc`,`lpc`文件夹建立软链接。

第一件事是重温构建声音的每个步骤，并查看是否有可以提高的地方。比如，可以调整 pitchmarking参数以适应你的声音。然后，尝试以下部分或全部变更。











