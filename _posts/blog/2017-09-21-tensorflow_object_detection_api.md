---
layout: post
title: window测试tensorflow object detection api
description: 深度学习实战
category: blog
---

## 0 注意

安装window版本的tensorflow时，如果tensorflow版本是1.3。需要安装 cudnn 6.0,但是貌似官网不让看了，windows端安装地址为： [windows cudnn6.0](http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/cudnn-8.0-windows10-x64-v6.0.zip)
window 端的为 [linux cudnn 6.0]( http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/cudnn-8.0-linux-x64-v6.0.tgz )

## 1 预备

### 1.1 Tensorflow Object Detection API 依赖：


+ Protobuf 2.6(最新版本是3.4，下面会提到如何安装)
+ Pillow 1.0 （从网站 [python第三方包下载网站](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy)）
+ lxml（从网站 [python第三方包下载网站](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy)）
+ tf Slim (`tensorflow/models`模块中包含了)
+ Jupyter notebook(如果不运行官网的网页测试就不需要)
+ Matplotlib
+ Tensorflow


### 1.2 下载model

下载链接： https://github.com/tensorflow/models

将`models/object_detection`拷贝到一个新工程目录`object_detection`下（工程名和代码目录都叫object_detection,工程名可以是其他）。我的目录结构如下：

![项目架构](/images/blog/tf_obj_detect_struct1.jpg)
 
之所以在弄两个object_detection，是要保留代码的引用逻辑，否则你要改一堆import 错误。而单独把object_detection抽出来是方便集成到其他工程里。

### 1.3 安装protoc

下载链接： [protoc](https://github.com/google/protobuf/releases)

![protoc下载](/images/blog/tf_obj_detect_download1.jpg)
 
我们在windows下使用，选择win32.下载后解压到某个目录下，解压 后的目录包含了`bin`目录：

![protoc](/images/blog/tf_obj_detect_win_proto_bin.jpg)

 为了避免夜长梦多，我直接把这个路径加入到window环境变量

![protoc环境变量](/images/blog/tf_obj_detect_win_var.jpg)
 
### 1.4 将proto文件生成对应的代码

虽然不太理解tensorflow的 `model/object_detection/protos/`目录下一堆`.proto`文件用处（proto貌似是谷歌文件传输数据格式），但是从安装过程大概可知，这些文件会被生成python文件。依赖的是一下命令：

```
protoc object_detection/protos/*.proto --python_out=.
```
注意该命令是在你的 `object_detection`文件夹的上一层目录下执行，默认是在`tensorflow/model`下。
 
![protoc前后](/images/blog/tf_obj_detect_proto_effect.jpg)
 
### 1.5 预知的错误

如果存在 

```
from nets import xxx

```
错误，是因为官网教程中提到的一句，将`slim`要加入到python环境变量中。

```
# From tensorflow/models/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

windows下没法完成这句，所以我们在需要用到nets的时候，把对应的网络（位于`\models\slim\nets`）复制过去即可。比如 `faster_rcnn_inception_resnet_v2_feature_extractor`开始的

```
from nets import inception_resnet_v2
```
这一句显然无法执行，我们可以替换为:

```
from object_detection.nets import inception_resnet_v2
```

从`\models\slim\nets`下将`nets`文件夹拷贝到`object_detect/object_detection`工程下。

![工程结构](/images/blog/tf_obj_detect_copy_net.jpg)
 
## 2 编写测试代码

测试代码基本复制自官方的 jupyter notebook脚本，名字为`object_detection_test.py` 

![测试demo](/images/blog/tf_obj_detect_test_code.jpg)
 
代码为：

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

import cv2  # add 20170825

cap = cv2.VideoCapture(0)  # add 20170825
sys.path.append("..")

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())

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

PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

# In[10]:

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:  # for image_path in TEST_IMAGE_PATHS:    #changed 20170825
            ret, image_np = cap.read()

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            cv2.imshow('object detection', cv2.resize(image_np, (1024, 800)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

```

注意，启动程序之前得准备个摄像头。

测试效果

![测试demo效果](/images/blog/tf_obj_detect_test_result.jpg)

参考 :http://blog.csdn.net/c20081052/article/details/77608954

