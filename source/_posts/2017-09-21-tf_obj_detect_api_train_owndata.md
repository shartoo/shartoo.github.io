---
layout: post
title: 使用tensorflow object detection api训练自己的数据
description: 深度学习实战
category: 深度学习
mathjax: true
---

## 一  数据准备

首先，我们有如下数据结构如下：

+ data
  + annotations:标注文件
     - txt：txt文本标注文件
     - xmls：xml格式标注文件
  + images：图像文件
  + config: 配置文件目录，下面有个当前数据集的 `.config` 配置文件。
  + tf_records：需要创建的一个目录，用于存储tensorflow将images转换为tf_records。
  + xx_label_map.pbtxt:分类名称对应的整型分类

### 1.1 images文件

images目录下的文件为：

![images目录](/images/blog/tf_obj_detect_own_iamges.jpg)

 ### 1.2  标注文件

xml标注文件类似：

![xml标注文件](/images/blog/tf_obj_detect_own_xml.jpg)

 txt标注文件可以不需要。

### 1.3  label_map.pbtxt文件

`xx_label_map.pbtxt`文件中的内容如下：

```
item {
  id: 1
  name: 'Abyssinian'
}

item {
  id: 2
  name: 'american_bulldog'
}

item {
  id: 3
  name: 'american_pit_bull_terrier'
}

```

### 1.4  创建tf_record文件

先创建一个`create_xx_tf_record.py`文件，单独用来处理训练数据。可以直接从object_detection工程下的`create_pacal_tf_record.py`（如果是每个图片只有一个分类，可以使用`create_pet_tf_record.py`）复制而来。

修改起始参数配置：

+ data_dir: 数据目录，包含了图片和标注的目录
+ output_dir:输出目录，图片转换为tf_record之后存储的位置
+ label_map_path:上面提到的xx_label_map.pbtxt

修改`dict_to_tf_example`

 参考你的标准xml文件，有些地方需要修改。
 
 ![dict_to_tf](/images/blog/tf_obj_detect_own_dict.jpg)

修改`main`

![修改main](/images/blog/tf_obj_detect_own_main.jpg)

 确保你的标注文件，图片目录对应的目录。标注文件目录下是否存在 `trainval.txt`文件是否存在，这个需要**自己生成**。我生成的列表（注意：没有带后缀）为：

![trainval文件](/images/blog/tf_obj_detect_own_trainval.jpg)
 
执行完之后会在对应目录下生成 tf_record文件。

### 1.5 创建 `.config` 配置文件

目录`tensorflow\models\object_detection\samples\configs`下有各种配置文件，当前工程使用的是  `faster_rcnn_inception_resnet_v2_robot.config`，将其修改为适应当前数据的配置。

主要修改了这些参数：

+ num_classes： 分类数目。视数据分类数目而定，当前数据集只有3个分类，修改为3
+ fine_tune_checkpoint：此处应该为空白，之前修改成github上下载的faster_rcnn的ckpt文件会导致无法训练的情况。
+ from_detection_checkpoint： 设置为true
+ num_steps: 训练步数。如果数据集较小，可以修改为较小。`pets`数据集包含7393张图片设置为20万次，当前数据集只有500张，设置为一万次应该差不多。可以在训练的时候查看loss增减情况来修改步数。

## 2 训练

训练时执行`train.py`即可。不过需要传入一些参数，可以使用官网的指定方式：

```
python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --train_dir=${PATH_TO_TRAIN_DIR}
```
我在pycharm下运行，所以在Run->configigure里面加入参数即可。需要指定的参数是：

+ pipeline_config_path:上面提到的`.config`配置文件
+ train_dir: 训练模型过程中保存的ckpt文件（tensorflow的权重文件）

```
--logtostderr --pipeline_config_path=D:/data/robot_auto_seller/config/faster_rcnn_inception_resnet_v2_robot.config --train_dir=D:/data/robot_auto_seller/tf_ckpt
```

训练完成之后，大概的效果如下：

![训练效果](/images/blog/tf_obj_detect_own_train_result.jpg)

如果训练得当，应该可以用tensorboard查看训练参数变化：

![tensorboard](/images/blog/tf_obj_detect_own_tensorboard_cmd.jpg)

打开浏览器中的： http://localhost:6006/#scalars

![tensorboard2](/images/blog/tf_obj_detect_own_tensorboard2.jpg)

##  3 转换权重文件

训练完成之后的权重文件大概是会包含如下文件:

+ model.ckpt-${CHECKPOINT_NUMBER}.data-00000-of-00001,
+ model.ckpt-${CHECKPOINT_NUMBER}.index
+ model.ckpt-${CHECKPOINT_NUMBER}.meta

我生成的大概为：

![ckpt文件](/images/blog/tf_obj_detect_own_ckpt.jpg)

 这些文件无法直接使用，`eval.py` 所使用的权重文件是`.pb`。需要做一步转换，object_detection工程中已经包含了该工具`export_inference_graph.py`，运行指令为：

```
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix ${TRAIN_PATH} \
    --output_directory output_inference_graph.pb
```

+ pipeline_config_path :pipeline的配置路径，使用的是上面训练所使用的`.config`文件
+ trained_checkpoint_prefix :上一步保存tensorflow的权重文件ckpt的。精确到step数目，比如为`xxx/model.ckpt-8876`
+ output_directory ：最终输出的可以用来做inference得文件（到具体文件名称）

我的脚本为：

```
--input_type image_tensor --pipeline_config_path D:/data/aa/config/faster_rcnn_inception_resnet_v2_robot.config --trained_checkpoint_prefix D:/data/aa/tf_ckpt/model.ckpt-6359  --output_directory  D:/data/aa/robot_inference_graph
```

生成的效果为：

![pb文件](/images/blog/tf_obj_detect_own_graph.png)


 ## 4  预测

预测代码为：

```
# coding: utf-8
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2  
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

cap = cv2.VideoCapture(0)
PATH_TO_CKPT = 'D:/data/aa/robot_inference_graph/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('D:/data/aa', 'robot_label_map.pbtxt')
NUM_CLASSES = 3

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# # Detection
PATH_TO_TEST_IMAGES_DIR = 'D:/data/aa/images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, '000{}.jpg'.format(i)) for i in range(109, 115)]
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        #while True:  # for image_path in TEST_IMAGE_PATHS:    #changed 20170825
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        for image_path in TEST_IMAGE_PATHS:
            image = Image.open(image_path)
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            print(boxes)
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            plt.figure(figsize=IMAGE_SIZE)
            cv2.imwrite('D:/data/robot_auto_seller/'+os.path.basename(image_path),image_np)
            plt.imshow(image_np)

```
此检测过程有两个版本。一个版本是开启摄像头检测，一个版本是直接检测图片。上面这部分代码是检测图片的。修改部分为

+ PATH_TO_CKPT ： 训练生成的`.pb`权重文件（上一步转换之后的结果）
+ PATH_TO_LABELS ：标签和分类(int)对应关系配置文件。第一步中设置的。
+ NUM_CLASSES ： 分类数。当前数据集是3个分类
+ PATH_TO_TEST_IMAGES_DIR ：需要检测的图片的路径。
TEST_IMAGE_PATHS ： 需要检测的图片列表。

 使用摄像头检测的例子放在附件中了。




参考：
[tensorflow 官方教程](https://github.com/tensorflow/models/blob/master/object_detection/object_detection_tutorial.ipynb)
[浣熊检测（英文）](https://medium.com/towards-data-science/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)
[tensorflow 生成pb文件](http://blog.csdn.net/qq_20373723/article/details/77838545)





