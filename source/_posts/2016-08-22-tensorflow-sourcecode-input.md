---
layout: post
title: tensorflow:CIFAR-10 图像处理源码.input.py
description: tensorflow
category: blog
---

# 源码解读


```
  from __future__ import absolute_import
  from __future__ import division
  from __future__ import print_function

  import os

  from six.moves import xrange  # pylint: disable=redefined-builtin
  import tensorflow as tf

  # 处理当前尺寸大小的图像，注意这与CIFAR-10图像的32x32尺寸不同。如果更新了这个数，那么整个模型架构都需要改变，并且模型需要重新训练
  IMAGE_SIZE = 24

  # 描述 CIFAR-10 数据集的全局常量
  NUM_CLASSES = 10
  NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
  NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

```

  '''
   作用： 读取并解析CIFAR10数据文件抽样数据。  
   注意:  如果需要N路并行读取，N次调用此函数即可。它会返回N个读取不同文件和位置的独立的reader

    @param  filename_queue  要读取的文件名队列
    @return 某个对象，具有以下字段:
            height: 结果中的行数 (32)
            width:  结果中的列数 (32)
            depth:  结果中颜色通道数(3)
            key:    一个描述当前抽样数据的文件名和记录数的标量字符串
            label:  一个 int32类型的标签，取值范围 0..9.
            uint8image: 一个[height, width, depth]维度的图像数据
  '''
```
  def read_cifar10(filename_queue):
    class CIFAR10Record(object):
      pass
    result = CIFAR10Record()

    # CIFAR-10图像数据集维度
    # 输入格式详见 http://www.cs.toronto.edu/~kriz/cifar.html
    label_bytes = 1  # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # 每行记录由 图像+标签组成 ，并且每行的长度固定
    record_bytes = label_bytes + image_bytes
    # 读取一行记录，从filename_queue队列中获取文件名。CIFAR-10格式没有header和footer。默认设置为0
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # 将一个长度为record_bytes的字符串转换为uint8的向量
    record_bytes = tf.decode_raw(value, tf.uint8)
    # 第一个字节代表了标签，将其类型由 uint8转换为 int32
    result.label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
    # 标签之后的字节代表了图像数据，并且将其维度从[depth * height * width]转换为 [depth, height, width].
    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                             [result.depth, result.height, result.width])
    # 将 [depth, height, width] 转换为[height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result
```


   作用： 构建一队列的批量图像和标签

   @param  image :             维度为[height, width, 3]的3D张量
   @param  label :             图像标签
   @param  min_queue_examples: int32类型,从提供批量样本的队列中最小抽样数量.
   @param  batch_size:         每一批量的图像数量Number of images per batch.
   @param  shuffle:            boolean类型，决定是否使用混排队列

   @return
          images:              图像. 维度为 [batch_size, height, width, 3] 的4D张量
          labels:              一维标签，尺寸为 [batch_size]

```
    def _generate_image_and_label_batch(image, label, min_queue_examples,
                                      batch_size, shuffle):

    # 创建一个混排样本的队列，然后从样本队列中读取 'batch_size'数量的 images + labels数据（每个样本都是由images + labels组成）
    num_preprocess_threads = 16
    if shuffle:
      images, label_batch = tf.train.shuffle_batch(
          [image, label],
          batch_size=batch_size,
          num_threads=num_preprocess_threads,
          capacity=min_queue_examples + 3 * batch_size,
          min_after_dequeue=min_queue_examples)
    else:
      images, label_batch = tf.train.batch(
          [image, label],
          batch_size=batch_size,
          num_threads=num_preprocess_threads,
          capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.image_summary('images', images)

    return images, tf.reshape(label_batch, [batch_size])



  '''
  作用： 使用Reader操作构建扭曲的输入(图像)用作CIFAR训练

  @param  data_dir:   CIFAR-10数据目录
          batch_size: 每一批量的图像数
  @Returns:
        images: Images. 尺寸为 [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] 的4D张量
        labels: Labels. 大小为[batch_size] 的一维张量
  '''
  def distorted_inputs(data_dir, batch_size):

    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]
    for f in filenames:
      if not tf.gfile.Exists(f):
        raise ValueError('Failed to find file: ' + f)

    #创建一个先进先出的文件名队列，文件阅读器需要它来读取数据
    filename_queue = tf.train.string_input_producer(filenames)
    # 从文件名队列中读取样本
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE
    # 用于训练神经网络的图像处理，注意对图像进行了很多随机扭曲处理
    # 随机修建图像的某一块[height, width]区域
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
    #随机水平翻转图像
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # 由于这些操作都是不可累积的，考虑随机这些操作的顺序
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # 减去均值并处以像素的方差 (标准化)
    float_image = tf.image.per_image_whitening(distorted_image)

    # 确保随机混排有很好的混合性
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)

    # 通过构建一个样本队列来生成一批量的图像和标签
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)

  '''
    作用： 使用Reader ops操作构建CIFAR评估的输入

    @param:
           eval_data: boolean类型, 是否使用训练或评估数据集
           data_dir: CIFAR-10数据目录.
      batch_size: 每一批的图像数量
    Returns:
          images: Images. 尺寸为[batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]的4D张量.
          labels: Labels. 大小为[batch_size]的一维张量.
  '''
  def inputs(eval_data, data_dir, batch_size):

    if not eval_data:
      filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                   for i in xrange(1, 6)]
      num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
      filenames = [os.path.join(data_dir, 'test_batch.bin')]
      num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
      if not tf.gfile.Exists(f):
        raise ValueError('Failed to find file: ' + f)

    #创建一个先进先出的文件名队列，文件阅读器需要它来读取数据
    filename_queue = tf.train.string_input_producer(filenames)

    # 从文件名队列中读取抽样
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # 用于做评估的图像处理
    # 裁减图像的中心 [height, width] Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           width, height)

    # 减去均值并处以像素的方差（标准化）
    float_image = tf.image.per_image_whitening(resized_image)

    # 确保良好的随机性
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=False)
```
