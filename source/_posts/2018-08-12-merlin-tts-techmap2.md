---
layout: post
title: merlin语音合成讲义二：如何构建系统之数据准备
description: 语音
category: 语音
---

一 前端部分


![TTS merlin技术路线](/images/blog/merlin_tts_tch2_1.png)

如果存在已经训练好的前端工具，我们可以使用已经有的工具`Festival`,`MaryTTS`,`sSpeak`
如果我们没有标注数据，我们可以使用`Ossian`

### 1.1 Ossian Toolkit

+ 使用训练数据，可以使用最少的speech+text
 - 句子或段落对齐
+ 可以利用用户的任何额外数据
+ 提供`前端模块`和`胶水`来组合，Merlin DNNs

我们将展示
+ Ossian如何与Merlin结合来构建一个`Swahili`声音而不需要任何语言专家，只需要转录语音
+ 引入Ossian的某些思路来管理，而不需要标注

#### 1.1.1  Ossian ：Training Data
我们仅需要UTF-8格式的文本和语音，同时匹配 `句子 除以 段落` 尺寸的chunks

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_2.png)

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_3.png)

**utt文件**

```
<utt text ="Khartoum imejitenga na mzozo huo." waveform="./wav/pm_n2336.wav"  utterance_name="pm_n2236">
```
其中`utterance_name="pm_n2236"`是一个XML格式的发生结构，在训练集语料库为每个句子构建的。
其内容如下：
```
<utt text="Khartoum imejitenga na mzozo huo." waveform="./wav/pm_n2236.wav"
utterance_name="pm_n2236" processors_used=",word_splitter">
<token text="_END_" token_class="_END_"/>
<token text="Khartoum" token_class="word"/>
<token text=" " token_class="space"/>
<token text="imejitenga" token_class="word"/>
<token text=" " token_class="space"/>
<token text="na" token_class="word"/>
<token text=" " token_class="space"/>
<token text="mzozo" token_class="word"/>
<token text=" " token_class="space"/>
<token text="huo" token_class="word"/>
<token text="." token_class="punctuation"/>
<token text="_END_" token_class="_END_"/>
</utt> 
```
unicode字符属性用以无关语言的正则表达式来tokenise文本
正则表达式


![TTS merlin技术路线](/images/blog/merlin_tts_tch2_4.png)

同时unicode用来将tokens分类为单词，空格和标点。

### 1.1.2 词性标注(POS Tagging)


![TTS merlin技术路线](/images/blog/merlin_tts_tch2_5.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_6.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_7.png)


#### 4.1.3 分布式词向量作为POS(Part Of Speech) tag 替代品


![TTS merlin技术路线](/images/blog/merlin_tts_tch2_8.png)
分别来看
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_9.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_10.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_11.png)

将所有词映射到词向量空间
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_12.png)

将词向量替代POS
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_13.png)

以字母替代音素的标注文件
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_14.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_15.png)



### 4.2 强制对齐和 静音检测

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_16.png)

### 4.3 phrasing(短语)
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_17.png)



## 5 语言特征工程：使用XPATHS 做flatten

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_18.png)
对应的详细标注
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_19.png)

## 6 语言特征和工程


![TTS merlin技术路线](/images/blog/merlin_tts_tch2_20.png)

### 6.1 语言特征工程：flatten到上下文依赖的音素

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_21.png)


注意看左下角，`ao-th+er`，当前音素为`th`，其前缀为`ao`,后缀为`er`。进一步

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_22.png)

得到一个完整的音素标注。再进一步

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_23.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_24.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_25.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_26.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_27.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_28.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_29.png)

详细解释上面的标注文件

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_30.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_31.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_32.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_33.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_34.png)

一个句子的完整标注文件
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_35.png)


### 6.3 将每个上下文依赖的音素编码为向量

**示例**：使用一个长度为1-40二进制码来编码quinphone

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_36.png)

对应的二进制码

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_37.png)

开始编码，以头`sil`开始：

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_38.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_39.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_40.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_41.png)


### 6.4 语言特征工程：上采样到声学帧率(framerate)

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_42.png)

### 6.5 语言特征工程：添加音素内(within-phone)位置特征

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_43.png)

## 7 时域到底来源于哪里

先看脚本
02_prepare_labels.sh
```
# alignments can be state-level (like HTS) or phone-level
if [ "$Labels" == "state_align" ]
./scripts/run_state_aligner.sh $wav_dir $inp_txt $lab_dir $global_config_file
elif [ "$Labels" == "phone_align" ]
./scripts/run_phone_aligner.sh $wav_dir $inp_txt $lab_dir $global_config_file
# the alignments will be used to train the duration model later
cp -r $lab_dir/label_$Labels $duration_data_dir
# and to upsample the linguistic features to acoustic frame rate
# when training the acoustic model
cp -r $lab_dir/label_$Labels $acoustic_data_dir
```
run_state_aligner.sh
```
#do prepare full-contextual labels without timestamps
echo "preparing full-contextual labels using Festival frontend..."
bash ${WorkDir}/scripts/prepare_labels_from_txt.sh $inp_txt $lab_dir $global_config_file $train
# do forced alignment using HVite from HTK
python $aligner/forced_alignment.py
```
forced_alignment.py
```
aligner = ForcedAlignment()
aligner.prepare_training(file_id_list_name, wav_dir, lab_dir, work_dir, multiple_speaker)
aligner.train_hmm(7, 32)
aligner.align(work_dir, lab_align_dir)
```

## 8 声学特征抽取和工程
### 8.1 为何我们使用声学特征抽取-波形waveform
+ 音素 a: 的波形

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_44.png)

+ 音素a:的magnitude spectrum（幅度频谱）

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_45.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_46.png)


### 8.2 术语

+ Spectral Envelope：频谱封装

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_47.png)

+ F0

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_48.png)


+ Aperiodic energy：非周期能量

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_49.png)

### 8.5 典型声码器 WORLD
+ 语音特征
  - Spectral Envelope(使用CheapTrick评估)
  - F0：使用DIO评估
  - Band aperiodicties：使用D4C评估

#### 8.5.1 spectral envelope 评估

+ Hanning窗长度3T0

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_50.png)

+ 使用一个**长度为 2/3 F0**移动平均过滤器

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_51.png)
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_52.png)

+ 使用一个**长度为 2 F0**移动平均过滤器

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_53.png)

+ $SpEnv= q_0logSp(F)+q1logSp(F+F0)+q1logSp(F-F0)$

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_54.png)

#### 8.5.3 F0 评估

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_55.png)

#### 8.5.4 band aperiodicities

+ 能量和非能量之间的比率，对固定频率bands取平均
+ 比如： 总功率/sine 波形功率(total power /sine wave power)
+ 示例中，此比例为
 - 最低band： a
 - 更多band： b
 - 最高band： c
 
![TTS merlin技术路线](/images/blog/merlin_tts_tch2_56.png)


### 8.6 声学特征工程

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_57.png)

原始声学特征与实际使用的声学特征

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_58.png)

抽取一部分来分析

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_59.png)

再细致来看

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_60.png)

处理步骤如下

![TTS merlin技术路线](/images/blog/merlin_tts_tch2_61.png)

可以直接运行脚本`03_prepare_acoustic_features.sh`得到
```
python ${MerlinDir}/misc/scripts/vocoder/${Vocoder,,}/
extract_features_for_merlin.py ${MerlinDir} ${wav_dir} ${feat_dir} $SamplingFreq
```
`extract_features_for_merlin.py`

```
# tools directory
world = os.path.join(merlin_dir, "tools/bin/WORLD")
sptk = os.path.join(merlin_dir, "tools/bin/SPTK-3.9")
if fs == 16000:
nFFTHalf = 1024
alpha = 0.58
elif fs == 48000:
nFFTHalf = 2048
alpha = 0.77
mcsize=59
world_analysis_cmd = "%s %s %s %s %s" % (os.path.join(world, 'analysis'), \
filename,
os.path.join(f0_dir, file_id + '.f0'), \
os.path.join(sp_dir, file_id + '.sp'), \
os.path.join(bap_dir, file_id + '.bapd'))
os.system(world_analysis_cmd)
### convert f0 to lf0 ###
sptk_x2x_da_cmd = "%s +da %s > %s" % (os.path.join(sptk, 'x2x'), \
extract_features_for_merlin.py
os.path.join(f0_dir, file_id + '.f0a'), \
os.path.join(sptk, 'sopr') + ' -magic 0.0 -LN -MAGIC
-1.0E+10', \
os.path.join(lf0_dir, file_id + '.lf0'))
os.system(sptk_x2x_af_cmd)
### convert sp to mgc ###
sptk_x2x_df_cmd1 = "%s +df %s | %s | %s >%s" % (os.path.join(sptk, 'x2x'), \
os.path.join(sp_dir, file_id + '.sp'), \
os.path.join(sptk, 'sopr') + ' -R -m 32768.0', \
os.path.join(sptk, 'mcep') + ' -a ' + str(alpha
' -m ' + str(
mcsize) + ' -l ' + str(
nFFTHalf) + ' -e 1.0E-8 -j 0 -f 0.0 -q 3 ',
os.path.join(mgc_dir, file_id + '.mgc'))
os.system(sptk_x2x_df_cmd1)
### convert bapd to bap ###
sptk_x2x_df_cmd2 = "%s +df %s > %s " % (os.path.join(sptk, "x2x"), \
os.path.join(bap_dir, file_id + ".bapd"), \
os.path.join(bap_dir, file_id + '.bap'))
os.system(sptk_x2x_df_cmd2)
```
























