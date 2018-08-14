---
layout: post
title: merlin语音合成讲义三：系统回归
description: 语音
category: blog
---


## 1 概览
前馈神经网络
+ 概念上直白的
+ 对每个输入帧frame
  - 执行回归得到对应的输出特征
+ 为避免更广(wider)的输入上下文，可以简单的将几个frame堆叠
+ 需要注意的是：语言特征已经跨越(span)了几个时间尺度(timescale)

### 1.1 方向
+ 前馈架构
  - 没有记忆
+ 简单的循环神经网络
+ 梯度消失现象
+ LSTM神经元解决了梯度消失现象（其他类型的可能存在）

**但是**
+ 输入和输出有相同的帧率(frame rate)
+ 需要一个额外的时钟或者对齐机制来对输入做上采样

### 1.2 sequence-to-sequence
+ 下一步是，集成对齐机制到网络内部
+ 当前：输入序列长度可能与输出序列长度不一致

+ 例如：
   - 输入：上下文依赖的音素序列
  - 输出：声学帧(对于声码器vocoder)

+ 概念上
  - 读取整个输入序列；使用一个固定长度的表征来记忆
  - 给定表征，写输出序列

+ encoder（编码器）
+ 是一个循环神经网络，读入整个输入序列，然后用固定长度表征来summarises或者memorises他们。

![TTS merlin技术路线](/images/blog/merlin_tts_tch3_1.png)

+ encoder和decoder


![TTS merlin技术路线](/images/blog/merlin_tts_tch3_2.png)

### 1.3 sequence-to-sequence中的对齐
+ 基本模型，输入和输出之间没有对齐
+ 通过加入注意力模型来获得更好结果
   - decoder可以接近输入序列
  - decoder也可以在前一个时间步(time step)接近其输出
+ 对齐像ASR模型。但是用声码器(vocoder)来做ASR效果不好
   - 因而我们期望通过使用ASR样式的声学特征(仅仅是模型的对齐部分)来获得更好效果

`04_prepare_conf_files.sh`
```
echo "preparing config files for acoustic, duration models..."
./scripts/prepare_config_files.sh $global_config_file
echo "preparing config files for synthesis..."
./scripts/prepare_config_files_for_synthesis.sh $global_config_file
```

`05_train_duration_model.sh`
```
./scripts/submit.sh ${MerlinDir}/src/run_merlin.py $duration_conf_file
```
`config files`
```
[DEFAULT]
Merlin: <path to Merlin root directory>
TOPLEVEL: <path where experiments are created>
[Paths]
# where to place work files
work: <path where data, log, models and generated data are stored and created>
# where to find the data
data: %(work)s/data
# where to find intermediate directories
inter_data: %(work)s/inter_module
# list of file basenames, training and validation in a single list
file_id_list: %(data)s/file_id_list.scp
test_id_list: %(data)s/test_id_list.scp
in_mgc_dir: %(data)s/mgc
in_bap_dir : %(data)s/bap
[Labels]
enforce_silence: False
silence_pattern: ['*-sil+*']
# options: state_align or phone_align
label_type: state_align
label_align: <path to labels>
question_file_name: <path to questions set>
add_frame_features: True
# options: full, coarse_coding, minimal_frame, state_only, frame_only, none
subphone_feats: full
[Outputs]
# dX should be 3 times X
mgc : 60
dmgc : 180
bap : 1
dbap : 3
lf0 : 1
dlf0 : 3
[Waveform]
[Outputs]
# dX should be 3 times X
mgc : 60
dmgc : 180
bap : 1
dbap : 3
lf0 : 1
dlf0 : 3
[Waveform]
test_synth_dir: None
# options: WORLD or STRAIGHT
vocoder_type: WORLD
samplerate: 16000
framelength: 1024
# Frequency warping coefficient used to compress the spectral envelope into MGC (or MCEP)
fw_alpha: 0.58
minimum_phase_order: 511
use_cep_ap: True
[Architecture]
switch_to_keras: False
hidden_layer_size : [1024, 1024, 1024, 1024, 1024, 1024]
hidden_layer_type : ['TANH', 'TANH', 'TANH', 'TANH', 'TANH', 'TANH']
model_file_name: feed_forward_6_tanh
#if RNN or sequential training is used, please set sequential_training to True.
sequential_training : False
dropout_rate : 0.0
batch_size : 256
# options: -1 for exponential decay, 0 for constant learning rate, 1 for linear decay
lr_decay : -1
learning_rate : 0.002
# options: sgd, adam, rprop
optimizer : sgd
warmup_epoch : 10
training_epochs : 25
[Processes]
# Main processes
AcousticModel : True
GenTestList : False
# sub-processes
NORMLAB : True
MAKECMP : True
NORMCMP : True
TRAINDNN : True
DNNGEN : True
GENWAV : True
CALMCD : True
```

`06_train_acoustic_model.sh`
```
./scripts/submit.sh ${MerlinDir}/src/run_merlin.py $acoustic_conf_file
```

`07_run_merlin.sh`
```
inp_txt=$1
test_dur_config_file=$2
test_synth_config_file=$3
echo "preparing full-contextual labels using Festival frontend..."
lab_dir=$(dirname $inp_txt)
./scripts/prepare_labels_from_txt.sh $inp_txt $lab_dir $global_config_file
echo "synthesizing durations..."
./scripts/submit.sh ${MerlinDir}/src/run_merlin.py $test_dur_config_file
echo "synthesizing speech..."
./scripts/submit.sh ${MerlinDir}/src/run_merlin.py $test_synth_config_file
```

## 2 设计选择：声学模型

+ 直白的方式：如果输入和输出**有相同长度并且是对齐的**
+ 前馈神经网络
+ 循环神经网络层
+ 不那么直白的方式：对非对齐输入和输出序列
   - 使用sequence-to-sequence
+ 唯一的实践限制是，是使用什么技术，比如Theano,Tensorflow

###  2.1 方向
+ 回归的输出是什么
   - 声学特征
  - 不是语音波形
所以还需要进一步
+ 生成波形
  - 输入时声学特征
  - 输出是语音波形

## 3 波形生成(waveform generator)

### 3.1 从声学(acoustic)特征回到原始声码器(vocoder)特征


![TTS merlin技术路线](/images/blog/merlin_tts_tch3_3.png)

### 3.2 WORLD：periodic excitation using a pulse train

+ 脉冲位置的计算
 - 声音分割：每一个fundamental period(基本周期)创建一个脉冲，T0。从F0计算T0，其中F0之前被声学模型预测得到
 - 非声音分割：固定频率 $T0=5ms$
### 3.3 WORLD:obtain spectral envelope at exact pulse locations, by interpolation(插值法在每个确定的脉冲位置获取频谱包络)

![TTS merlin技术路线](/images/blog/merlin_tts_tch3_4.png)

### 3.4 WORLD：重构周期性和非周期性的幅度频谱(magnitude spectra)


![TTS merlin技术路线](/images/blog/merlin_tts_tch3_5.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch3_6.png)

### 3.5 WORLD:生成波形

![TTS merlin技术路线](/images/blog/merlin_tts_tch3_7.png)

## 4 拓展
+ 混合语音合成
  - 使用Merlin来预测声学特征，使用Festival来做单元选取(select unit)
+ 声音转换
  - 输入语音而非文本
 - 训练数据是对齐的输入和输出语音（而不是音素标签和语音）
+ 讲话人调整
  - 增强输入
  - 调整隐藏层
  - 转换输出

### 4.1 经典单元选取

此处以音素单元为例，目标和join cost

![TTS merlin技术路线](/images/blog/merlin_tts_tch3_8.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch3_9.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch3_10.png)

### 4.2 独立特征形式(Independent Feature Formulation(IFF))目标损失

![TTS merlin技术路线](/images/blog/merlin_tts_tch3_11.png)

### 4.3 声学空间形式(Acoustic Space Formulation)目标损失

![TTS merlin技术路线](/images/blog/merlin_tts_tch3_12.png)

### 4.4  混合语音合成就像使用Acoustic Space Formulation目标损失的单元选取

![TTS merlin技术路线](/images/blog/merlin_tts_tch3_13.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch3_14.png)

### 4.5 混合语音合成就像：统计参数语音合成，使用声码器(vocoder)的替换

![TTS merlin技术路线](/images/blog/merlin_tts_tch3_15.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch3_16.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch3_17.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch3_18.png)

### 4.6 混合语音合成就像：同时对目标和join cost使用混合密度网络

![TTS merlin技术路线](/images/blog/merlin_tts_tch3_19.png)

### 7 声音转换

将源声转换为另外一个人的声音，而不改变声音内容

![TTS merlin技术路线](/images/blog/merlin_tts_tch3_19.png)

使用神经网络完成

![TTS merlin技术路线](/images/blog/merlin_tts_tch3_20.png)

### 7.1 输入和输出的声学特征的抽取和工程

![TTS merlin技术路线](/images/blog/merlin_tts_tch3_21.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch3_22.png)

### 7.2 输入输出的对齐

+ 从波形waveform中声学特征
+ 使用动态时间封装(Dynamic Time Wraping(DTW))

![TTS merlin技术路线](/images/blog/merlin_tts_tch3_23.png)

### 7.3 最简单的方法：对齐输入和输出特征+逐帧回归

![TTS merlin技术路线](/images/blog/merlin_tts_tch3_24.png)

### 7.4 当然，我们也可以用前馈神经网络做得更好

![TTS merlin技术路线](/images/blog/merlin_tts_tch3_25.png)

我们可以使用`Merlin/egs/voice_conversion/s1/`目录下的脚本完成这个工作

`03_align_src_with_target.sh`

```
src_feat_dir=$1
tgt_feat_dir=$2
src_aligned_feat_dir=$3
src_mgc_dir=$src_feat_dir/mgc
tgt_mgc_dir=$tgt_feat_dir/mgc
echo "Align source acoustic features with target acoustic features..."
python ${MerlinDir}/misc/scripts/voice_conversion/dtw_aligner_festvox.py ${MerlinDir}/tools
${src_feat_dir} ${tgt_feat_dir} ${src_aligned_feat_dir} ${bap_dim}
```
![TTS merlin技术路线](/images/blog/merlin_tts_tch3_26.png)

## 8 讲话人调整(Speaker Adaptation)

+ 只使用了目标讲话人一小段录音来创造一个新的声音

![TTS merlin技术路线](/images/blog/merlin_tts_tch3_27.png)

### 8.1 使用DNN方法的讲话人调整

+ 需要额外的输入特征
+ 应用转换（声音转换）到输出特征
+ 学习一个模型参数的修改(LHUC)
+ 共享层(hat swapping)
+ 在目标讲述人数据上fine-tuning整个模型

![TTS merlin技术路线](/images/blog/merlin_tts_tch3_28.png)

共享层和hot swapping

![TTS merlin技术路线](/images/blog/merlin_tts_tch3_29.png)















