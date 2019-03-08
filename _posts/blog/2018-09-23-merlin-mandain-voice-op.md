---
layout: post
title: merlin语音合成方案mandarin_voice操作步骤
description: 语音何晨
category: blog
---

### 0 概览
本文详细解释Merlin Mandarin_voice下脚本一步一步所做的事。

### 01_setup
脚本`merlin/egs/mandarin_voice/s1/01_setup.sh`

主要工作是创建一个目录，做好准备工作。主要创建了如下文件夹:

+ experiments

```
─ mandarin_voice(voice name)
    ├── acoustic_model
    │ ├── data
    │ ├── gen
    │ ├── inter_module
    │ ├── log
    │ └── nnets_model
    ├── duration_model
    │ ├── data
    │ ├── gen
    │ ├── inter_module
    │ ├── log
    │ └── nnets_model
    └── test_synthesis
        ├── gen-lab
        ├── prompt-lab
        ├── test_id_list.scp
        └── wav
```

+ database

```
 feats
│ ├── bap
│ ├── lf0
│ └── mgc
├── labels
│ └── label_phone_align
├── prompt-lab
│ ├── A11_0.lab
│ ├── A11_1.lab
│ ├── A11_2.lab
...
└── wav
    ├── A11_0.wav
    ├── A11_100.wav
    ├── A11_101.wav

```


将一些基本参数写入到`conf/global_setting.cfg`文件中


![merlinmandarin voice操作](/images/blog/merlin_mandarin_voice_op1.jpg)


**注意：一定要在setup.sh里面定义好train,valid,test的数量，不然修改global_config.cfg里面的值也没用。这三者相加的值要等于（duration_model/FileIdList下）file_id_list.scp总行数**

### 02_prepare_lab

需要两个参数：

+ lab_dir: 第一步中的标注目录 `database/labels`
+ prompt_lab_dir :第一步中生成的`database/prompt-lab`

#### 2.1 准备文件夹

+ 将 `database/labels`目录下的`lab_phone_align`下的lab文件分别复制到`experiments/mandarin_voice/duration_model/data`（时域模型）和`experiments/mandarin_voice/acoustic_model/data`（声学模型）下。【用于训练】

+ 将`database/prompt-lab`下的lab文件复制到`experiments/mandarin_voice/test_synthesis`下【用于测试（合成）】

#### 2.2 生成文件列表

+ 将`database/labels`目录下的`lab_phone_align`下的lab文件列表写入到`experiments/mandarin_voice/duration_model/FileIdList'和`experiments/mandarin_voice/acoustic_model/FileIdList'。并移除文件后缀【训练集文件列表】

+  将`database/prompt-lab`下的lab文件列表写入到`experiments/mandarin_voice/test_synthesis/test_id_list.scp`文件中，并移除文件后缀【用于合成语音的文本列表】

###  03_prepare_acoustic_feature

需要两个参数

+ **wav_dir**: 使用的是第一步中的`database/wav`，下面存放的是所有的wav音频文件
+ **feat_dir**:输出文件目录`database/feats`，是当前脚本输出的特征存放文件目录
#### 3.1 使用声码器抽取声学特征

使用`merlin/misc/scripts/vocoder/world/extract_features_for_merlin.py`脚本抽取，注意，其中的声码器可以是`WORLD`也可以是其他的，比如`straight`,`WORLD_2`。其实依然是在python中调用以下脚本：

```
world = os.path.join(merlin_dir, "tools/bin/WORLD")
sptk = os.path.join(merlin_dir, "tools/bin/SPTK-3.9")
reaper = os.path.join(merlin_dir, "tools/bin/REAPER")
```

生成的特征目录如下：

```
sp_dir = os.path.join(feat_dir, 'sp' )
mgc_dir = os.path.join(feat_dir, 'mgc')
ap_dir = os.path.join(feat_dir, 'ap' )
bap_dir = os.path.join(feat_dir, 'bap')
f0_dir = os.path.join(feat_dir, 'f0' )
lf0_dir = os.path.join(feat_dir, 'lf0')
```

如果我们使用world作为vocoder的话，会使用`misc/scripts/vocoder/world/extract_features_for_merlin.py`脚本，生成步骤其实是：
1. 直接从原始wav文件，使用`world analysis`抽取 `sp`,`bapd`特征。`straight`vocoder 会产生 `ap`,如果使用reaper会产生`f0`特征。
2. `f0`$\rightarrow$ `lf0`,`bapd`$\rightarrow$ `bap`,`sp`$\rightarrow$ `mgc`


#### 3.2 复制特征到声学特征目录下

将所有`feat_dir`下的所有文件,包括`sp`,`mgc`,`ap`,`bap`,`f0`,`lf0`复制到`experiments/mandarin_voice/acoustic_model/data`下。

## 04_prepare_conf_files

执行`./scripts/prepare_config_files.sh `

**duration相关配置**
+ 先从`merlin/misc/recipes/duration_demo.conf`复制一份到`conf/duration_mandarin_voice.conf`，并修改`conf/duration_mandarin_voice.conf`中的一些目录
  - MerlinDir
  - WorkDir
  - TOPLEVEL
  - FileIdList

+ 修改Label相关的配置项【Labels】
  - silence_pattern：修改为 ` ['*-sil+*']`
  - label_type:`state_align` 或 `phone_align`，修改之后为`phone_align`
  - label_align: 即配置音素对齐文件的目录`/experiments/mandarin_voice/duration_model/data/label_phone_align`
  - question_file_name:`/misc/questions/questions-mandarin.hed`问题集

+ 修改输出配置【Outputs】，label_type有`state_align` 或 `phone_align`，如果是`state_align`会在【outputs】处指定`dur=5`,如果是`phone_align`则指定`dur=1`
+ 神经网络的架构配置，如果当前声音文件是`demo`则修改`hidden_layer_size` 【architechture】
+ 修改训练、验证、测试数据数量。【data】
  - train_file_number: 200
  - valid_file_number: 25
  - test_file_number: 25

**acoustic相关配置**
+ 复制文件`conf/acoustic_mandarin_voice.conf`，修改变量，label配置都和duration相关配置一样。
+ 修改输出配置【outputs】
  - mgc
  - dmgc
  - bap
  - dbap
  - lf0
  - dlf0
+ 波形文件设置【waveform】
  - framelength
  - minimum_phase_order
  - fw_alpha
+ 其他的【architechture】和【data】都和duration相关配置一样。

执行`./scripts/prepare_config_files_for_synthesis.sh`配置测试（或合成）语音相关的参数。基本和上面的`./scripts/prepare_config_files.sh `一样，需要配置`duration`和`ascoustic`参数。新增了【Processes】

```
DurationModel: True
GenTestList: True
# sub-processes
NORMLAB: True
MAKEDUR: False
MAKECMP: False
NORMCMP: False
TRAINDNN: False
DNNGEN: True
CALMCD: False
```

### 05_train_duration_model

实际执行的是`./scripts/submit.sh   merlin/src/run_merlin.py   conf/duration_mandarin_voice.conf`

其中`./scripts/submit.sh`是theano相关参数的配置。

### 06_train_acoustic_model

训练声学模型，实际执行的是`./scripts/submit.sh   merlin/src/run_merlin.py   conf/acoustic_mandarin_voice.conf`


### 07_run_merlin

需要两个参数
+ test_dur_config_file: 语音合成的时域配置文件
+ test_synth_config_file:语音合成的

