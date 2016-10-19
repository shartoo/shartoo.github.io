---
layout: post
title: 深度学习：mxnet重新训练inceptionv3模型
description: deep learning实战
category: blog
---

## 代码和数据位置

 + 训练代码位置：/home/xiatao/code/bot/bot_second_round/fine_tune.py
 + 原始模型目录：/home/xiatao/model/mxnet/inceptionv3_orginal_model
 + 图片数据目录: /data/bot_img/bot_secondround/all
 + 图片名称与标签列表文件目录：/home/xiatao/model/mxnet/bot_round2_model_save

## 预备数据

 mxnet需要先生成训练集和测试集的文件列表，文件列表示例如下:

 ![列表文件示例](/images/blog/mxnet_finetune1.png)

 其中第一列为文件id,第二列为文件标签，第三列为文件名。根据这个列表文件生成mxnet所需的`.rec`文件，执行代码为:

 ```
  ~/mxnet/bin/im2rec temp.lst ../all_img/ test.rec resize=480 quality=90
 ```

 示例代码为:

 ```
 /home/xiatao/mxnet/bin/im2rec train_BackLight.txt /data/bot_img/bot_secondround/all/ train_backlight.rec resize=480 quality=90

 ```

## 训练代码

 不同的模型所使用的参数不一样，基本的训练过程都写在`fine_tune.py`这个代码中，我们只需要在执行时指定不同参数即可。

```
python ~/mxnet/example/image-classification/fine_tune_0923.py --data-dir    # 图片数据目录
--model-prefix ./model/Inception-3 #pre_train模型存放的路径，需要*.params文件和一个.json文件
--save-model-prefix ./save/   #此目录必须先建好，否则保存模型时会报错
--num-epochs 5 #总共要训练到第几个epoch
--load-epoch 1  #从编号为1的pre_trian模型开始训练
--gpus 0
--num-examples     # 训练样本数
--num-classes      # 分类数目，不同的分类任务，分类数不一样
--log-dir         #手动执行日志文件目录 /home/xiatao/mylogs/bot_logs
--log-file        # 日志名称
--train-dataset train_0923.rec
--val-dataset val_0923.rec
```

## 模型预测

预测代码：

```
python ~/mxnet/example/image-classification/0923_predict.py --prefix ./save0923/ --epoch 17 --path ../img_t6_val/
```
