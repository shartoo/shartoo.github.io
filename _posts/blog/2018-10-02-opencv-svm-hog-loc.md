---
layout: post
title: 使用HOG+SVM+滑窗+NMS完成目标定位分类
description: 图像处理
category: blog
---

## 0  概览
整个过程如下:

1. 数据标注
2. 抽取HOG
3. 训练SVM
4. 预测（滑窗，分类）
5. NMS

## 1 数据标注

使用的是`extract_feature_from_fasterrcnn_labeled.py`

本文直接使用了 YOLO的数据标注格式。YOLO的标注格式如下：

```
# 一个有9个物体，9个物体总共属于3个分类
9,3
# 分类yinliao的类别是0，xmin,ymin,xmax,ymax分别是56,103,261,317
56,103,261,317,0,yinliao
261,0,442,120,2,kele
465,16,645,180,2,kele
209,138,345,266,2,kele
```

根据此标注文件可以从图片中抠取目标物体的ROI。如下图，根据标注的文件可以抽取的ROI示例，左上角为提取的ROI：

![yolov123](/images/blog/hog_svm_loc1.jpg) 

## 2 抽取特征

### 2.1 抽取HOG特征

注意：

+ 我最开始使用的是 Opencv的`hog = cv2.HOGDescriptor`的方式，后来修改为`skimage.feature`。
+ hog特征如果想要固定长度的话，提取对象(ROI)必须也是固定长度的，所以需要做个resize。否则的话在训练时，会出现数据维度不一致的问题。我之前以为HOG会把任意尺寸的图像转换为相同长度的特征，发现并不是。提取完之后检查输出文件的大小是否一样。
+ 提取ROI的时候注意 y在前，x在后。比如```roi = gray[int(rec[1]):int(rec[3]),int(rec[0]):int(rec[2])]```,rec里面是顺序的(xmin,ymin,xmax,ymax)
+ 使用 `sklearn.externals.joblib.dump(hog特征，保存路径)`的方式存储提取的HOG特征

ROI和其HOG特征示例如下（左边为ROI，右边为对应的HOG）：

![yolov123](/images/blog/hog_svm_loc2.jpg) 

当前的做法是每次提取一个正样本，就生成5个或者10个负样本。使用`generate_neg_box`方法。

### 2.2 SURF特征

参考

 [opencv3.0+的 surf特征使用](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html)

[使用SVM对图像数据分类](http://www.dummies.com/programming/big-data/data-science/how-to-visualize-the-classifier-in-an-svm-supervised-learning-model/)

[opencv surf特征一些参数的解释](https://docs.opencv.org/3.4.0/d5/df7/classcv_1_1xfeatures2d_1_1SURF.html)

代码如下：

```
def extract_surf(img,pca = 100,visual =False):
    image = cv2.imread(im)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    surf = cv2.xfeatures2d.SURF_create(1000, extended=False)
    (kps, descs) = surf.detectAndCompute(image, None)
    if visual:
        print(len(kps),descs.shape)
        print("==" * 30)
        print(descs)
        img2 = cv2.drawKeypoints(image, kps, None, (255, 0, 0), 4)
        cv2.imshow("with surf feature", img2)
        cv2.waitKey(0)
    return kps, descs
```

+  SURF_create:第一个参数是Hessian矩阵阈值，此值越大，生成的特征点越少；第二个参数是extended是是否需要拓展，False时每个特征点维度是64，True时每个特征点维度是128.

+ detectAndCompute的结果：有两个值kps和descs。其实descs包含kps，descs是一个二维数组，行数即特征点数目（不固定），列数固定为64或128.如下图：


![yolov123](/images/blog/hog_svm_loc3.jpg) 


## 3 训练SVM

代码在 [classifier.py]()中。训练SVM很简单，如下：

![yolov123](/images/blog/hog_svm_loc4.jpg) 

过程为：

1. 添加 `(正样本，正标签)`和`(负样本，负标签)`.

2. 声明一个SVM分类器

3. 拟合SVM分类器

4. 保存模型

训练过程非常快。

## 4 预测(滑窗，分类)

测试过程相对麻烦。需要处理几个问题

1. 训练的时候的ROI都是固定尺寸的(做了resize)，但是测试的时候可能物体有**尺寸变化**，如何应对尺寸变换->对原图做图像**金字塔缩放**(会生成大约9张不同尺寸的图)
2. 如何在一张图中**搜索物体**->使用固定尺寸的**滑窗遍历**图像
3. 临近位置的ROI可能会预测**多个结果**，合并结果->**NMS**

代码`classifier.py`的`test_model`方法截图：

![yolov123](/images/blog/hog_svm_loc5.jpg) 
![yolov123](/images/blog/hog_svm_loc6.jpg) 

### 4.1 图像金字塔

注意生成图像金字塔的时候并没有使用`skimage.transform.pyramid_gaussian`的，因为这种方法对使用HOG特征不太友好。所以此代码中重写了`pyramid`



## 5 总结

1. 不能使用opencv的hog描述子来计算和训练，否则在预测的时候会出现数据维度或格式错误’
2. HOG特征的长度是跟图像的尺寸有关的，所以在计算HOG特征之前要统一resize到固定尺寸才行。虽然HOG特征计算时声称，只跟
3. 使用SVM做二分类的时候要注意，负样本可能需要多一点。不然在预测时会出现很多误判。我刚开始时使用另外一个分类的ROI作为负样本，事实表明效果很差，最后采取了随机在正样本周围取样，效果会变好一点。


## 代码

**classifier.py**

```
#!/usr/bin/env python
#encoding:utf-8
"""
@author:
@time:2017/3/19 11:08
"""
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import numpy as np
import glob
import os
import cv2
from skimage.transform import pyramid_gaussian
import imutils
import matplotlib.pyplot as plt
from skimage.io import imread
from extract_feature_from_fasterrcnn_labeled import get_hog
from skimage import feature
import time

def train_model(pos_feat_path,neg_feat_path,model_path):
    fds = []
    labels = []
    # Load the positive features
    for feat_path in glob.glob(os.path.join(pos_feat_path, "*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(1)

    # Load the negative features
    for feat_path in glob.glob(os.path.join(neg_feat_path, "*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(0)

    clf = LinearSVC()
    print("Training a Linear SVM Classifier")
    max_len = 0
    for fd in fds:
        if len(fd)>max_len:
            max_len =len(fd)
    for i in range(len(fds)):
        fd = fds[i]
        np.squeeze(fd[i],axis=0)
        if len(fd)<max_len:
            fds[i] = np.concatenate((fds[i],np.array([0]*(max_len-len(fds[i])))),axis=0)
    clf.fit(fds, labels)
    # If feature directories don't exist, create them
    if not os.path.isdir(os.path.split(model_path)[0]):
        os.makedirs(os.path.split(model_path)[0])
    joblib.dump(clf, model_path)
    print("Classifier saved to {}".format(model_path))

def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input image `image` of size equal
    to `window_size`. The first image returned top-left co-ordinates (0, 0)
    and are increment in both x and y directions by the `step_size` supplied.
    So, the input parameters are -
    * `image` - Input Image
    * `window_size` - Size of Sliding Window
    * `step_size` - Incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    where
    * x is the top-left x co-ordinate
    * y is the top-left y co-ordinate
    * im_window is the sliding window image
    '''
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def overlapping_area(detection_1, detection_2):
    '''
    Function to calculate overlapping area'si
    `detection_1` and `detection_2` are 2 detections whose area
    of overlap needs to be found out.
    Each detection is list in the format ->
    [x-top-left, y-top-left, confidence-of-detections, width-of-detection, height-of-detection]
    The function returns a value between 0 and 1,
    which represents the area of overlap.
    0 is no overlap and 1 is complete overlap.
    Area calculated from ->
    http://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
    '''
    # Calculate the x-y co-ordinates of the
    # rectangles
    x1_tl = detection_1[0]
    x2_tl = detection_2[0]
    x1_br = detection_1[0] + detection_1[3]
    x2_br = detection_2[0] + detection_2[3]
    y1_tl = detection_1[1]
    y2_tl = detection_2[1]
    y1_br = detection_1[1] + detection_1[4]
    y2_br = detection_2[1] + detection_2[4]
    # Calculate the overlapping Area
    x_overlap = max(0, min(x1_br, x2_br)-max(x1_tl, x2_tl))
    y_overlap = max(0, min(y1_br, y2_br)-max(y1_tl, y2_tl))
    overlap_area = x_overlap * y_overlap
    area_1 = detection_1[3] * detection_2[4]
    area_2 = detection_2[3] * detection_2[4]
    total_area = area_1 + area_2 - overlap_area
    return overlap_area / float(total_area)

def nms(detections, threshold=.5):
    '''
    This function performs Non-Maxima Suppression.
    `detections` consists of a list of detections.
    Each detection is in the format ->
    [x-top-left, y-top-left, confidence-of-detections, width-of-detection, height-of-detection]
    If the area of overlap is greater than the `threshold`,
    the area with the lower confidence score is removed.
    The output is a list of detections.
    '''
    if len(detections) == 0:
        return []
    # Sort the detections based on confidence score
    detections = sorted(detections, key=lambda detections: detections[2],
            reverse=True)
    # Unique detections will be appended to this list
    new_detections=[]
    # Append the first detection
    new_detections.append(detections[0])
    # Remove the detection from the original list
    del detections[0]
    # For each detection, calculate the overlapping area
    # and if area of overlap is less than the threshold set
    # for the detections in `new_detections`, append the
    # detection to `new_detections`.
    # In either case, remove the detection from `detections` list.
    for index, detection in enumerate(detections):
        for new_detection in new_detections:
            if overlapping_area(detection, new_detection) > threshold:
                del detections[index]
                break
        else:
            new_detections.append(detection)
            del detections[index]
    return new_detections


def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image

def test_model(model_path,image):
    im = cv2.cvtColor(cv2.imread(image), cv2.COLOR_RGB2GRAY)
    min_wdw_sz = (50, 100)
    step_size = (30, 30)
    downscale = 1.25
    visualize_det = True
    # Load the classifier
    clf = joblib.load(model_path)
    visualize_test = True
    # List to store the detections
    detections = []
    # The current scale of the image
    scale = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter('D:/data/test_carmera/svm/hog_svm_slide_wid.avi', -1, 20.0, (480,600))

    # Downscale the image and iterate
    #for im_scaled in pyramid_gaussian(im, downscale=downscale):
    for (i, im_scaled) in enumerate(pyramid(im, scale=1.25)):
        # This list contains detections at the current scale
        cd = []
        # If the width or height of the scaled image is less than
        # the width or height of the window, then end the iterations.
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break
        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
               continue
            #Calculate the HOG features
            #fd = get_hog(im_window)
            #im_window = imutils.auto_canny(im_window)
            im_window =  cv2.resize(im_window,(200,250))
            fd = feature.hog(im_window, orientations=9, pixels_per_cell=(10, 10),cells_per_block=(2, 2), transform_sqrt=True)
            print(fd.shape)
            if len(fd)>1:
                #fd = np.transpose(fd)
                fd = fd.reshape(1, -1)
                pred = clf.predict(fd)
                print("prediction:\t ",pred)
                if pred == 1:# and clf.decision_function(fd)>1:
                    print("Detection:: Location -> ({}, {})".format(x, y))
                    print("Scale ->  {} | Confidence Score {} \n".format(scale, clf.decision_function(fd)))
                    detections.append((x, y, clf.decision_function(fd),
                                   int(min_wdw_sz[0] * (downscale ** scale)),
                                   int(min_wdw_sz[1] * (downscale ** scale))))

                    cd.append(detections[-1])
                # If visualize is set to true, display the working
                # of the sliding window
                if visualize_det:
                    clone = im_scaled.copy()
                    for x1, y1, _, _, _ in cd:
                        # Draw the detections at this scale
                        cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 +
                                                        im_window.shape[0]), (0, 0, 0), thickness=2)
                    cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y +
                                                  im_window.shape[0]), (255, 255, 255), thickness=2)
                    cv2.imshow("Sliding Window in Progress", clone)
                    out_img_tempt= cv2.resize(clone,(480,600))
                    video_out.write(out_img_tempt)
                    cv2.waitKey(10)
        # Move the the next scale
        scale += 1

    # Display the results before performing NMS
    clone = im.copy()
    for (x_tl, y_tl, _, w, h) in detections:
        # Draw the detections
        cv2.rectangle(im, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 0), thickness=2)
    cv2.imshow("Raw Detections before NMS", im)
    cv2.waitKey()

    # Perform Non Maxima Suppression
    detections = nms(detections, 0.5)

    # Display the results after performing NMS
    for (x_tl, y_tl, _, w, h) in detections:
        # Draw the detections
        cv2.rectangle(clone, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 0), thickness=2)
    cv2.imshow("Final Detections after applying NMS", clone)
    out_img_tempt = cv2.resize(clone,(480,600))
    #out_img_tempt[0:clone.shape[0], 0:clone.shape[1]] = clone[:, :]
    video_out.write(out_img_tempt)
    cv2.waitKey()
    video_out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    hog_svm_model = r"D:\data\imgs\hog_svm_model_kele.model"
    pos_feat_path = r"D:\data\imgs\hog_feats\2"
    neg_feat_path = r"D:\data\imgs\hog_feats\2_neg"

    # hog_svm_model = r"D:\data\test_carmera\svm\hog_svm_model.model"
    # pos_feat_path = r"D:\data\test_carmera\svm\hog_feats\0"
    # neg_feat_path = r"D:\data\test_carmera\svm\hog_feats\0_neg"

    train_model(pos_feat_path,neg_feat_path,hog_svm_model)
    test_img = "D:/data/test_carmera/svm/images/frmaes_2.jpg"
    test_img1 = r"D:\data\imgs\images\0b24fb0ee9947292ffbb88c6e7c22a08.jpg"
    test_model(hog_svm_model,test_img1)

    # gray = cv2.cvtColor(cv2.imread(test_img), cv2.COLOR_BGR2GRAY)
    # edged = imutils.auto_canny(gray)
    # cv2.imshow("edges",edged)
    # # find contours in the edge map, keeping only the largest one which
    # # is presumed to be the car logo
    # (a,cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # c = max(cnts, key=cv2.contourArea)
    # # extract the logo of the car and resize it to a canonical width
    # # and height
    # (x, y, w, h) = cv2.boundingRect(c)
    # logo = gray[y:y + h, x:x + w]
    # cv2.imshow("rect",logo)
    # cv2.waitKey(0)
    # t0 = time.time()
    # clf_type = 'LIN_SVM'
    # fds = []
    # labels = []
    # num = 0
    # total = 0
    # for feat_path in glob.glob(os.path.join(train_feat_path, '*.feat')):
    #     data = joblib.load(feat_path)
    #     fds.append(data[:-1])
    #     labels.append(data[-1])
    # if clf_type is 'LIN_SVM':
    #     clf = LinearSVC()
    #     print("Training a Linear SVM Classifier.")
    #     clf.fit(fds, labels)
    #     # If feature directories don't exist, create them
    #     # if not os.path.isdir(os.path.split(model_path)[0]):
    #     #     os.makedirs(os.path.split(model_path)[0])
    #     # joblib.dump(clf, model_path)
    #     # clf = joblib.load(model_path)
    #     print("Classifier saved to {}".format(model_path))
    #     for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
    #         total += 1
    #         data_test = joblib.load(feat_path)
    #         data_test_feat = data_test[:-1].reshape((1, -1))
    #         result = clf.predict(data_test_feat)
    #         if int(result) == int(data_test[-1]):
    #             num += 1
    #     rate = float(num)/total
    #     t1 = time.time()
    #     print('The classification accuracy is %f'%rate)
    #     print('The cast of time is :%f'%(t1-t0))
    #

```

**extract_feature_from_fasterrcnn_labeled.py**

```
# -*- coding:utf-8 -*-
'''
extract image hog feature from image labeled  for fasterrcnn
'''
import os
import cv2
import numpy as np
from sklearn.externals import joblib
from skimage import feature,exposure
from config import *
import imutils
import time
import random

def read_label_info_from_file(txt_file):
    '''
    read label information from txt file.
    content of file should looks like:
            9,3
            56,103,261,317,0,yinliao
            261,0,442,120,2,kele
            ...
    :param txt_file:
    :return:
    '''
    id_and_rect = {}
    with open(txt_file,'r') as label_info:
        lines = label_info.readlines()
        for line in lines[1:]:
            infos = line.split(",")
            xmin,ymin,xmax,ymax,class_id = int(infos[0].strip()),int(infos[1].strip()),int(infos[2].strip()),int(infos[3].strip()),str(infos[4].strip())
            if not class_id  in id_and_rect:
                id_and_rect[class_id] = [(xmin,ymin,xmax,ymax)]
            else:
                id_and_rect[class_id].append([xmin, ymin, xmax, ymax])
    return id_and_rect

def generate_neg_box(im_width,im_height,pos_boxes,times = 5):
    '''
        generate negative roi from image. negative roi should not in pos_boxes

    :param im_height:   height of original image
    :param im_width:    width of original image
    :param pos_boxes:   positive roi boxes(xmin,ymin,xmax,ymax)
    :param times:       times of negative vs positive boxes
    :return:            a list of negative roi [(xmin,ymin,xmax,ymax),...]
    '''
    neg_boxes = []
    min_size = min(im_width,im_height)
    mask = np.ones((im_width,im_height))
    for (xmin,ymin,xmax,ymax) in pos_boxes:
        mask[xmin:xmax,ymin:ymax] = 0

    for (xmin,ymin,xmax,ymax) in pos_boxes:
        tmp_width =  xmax-xmin
        tmp_height = ymax-ymin
        # get $times times of negative boxes for every positive box
        for _ in range(times):
            flag = True
            neg_x_min = 0
            neg_y_min = 0
            start_time = time.time()
            while flag:
                neg_x_min = random.randint(0,min_size-tmp_width)
                neg_y_min = random.randint(0,min_size-tmp_height)
                tmp_rect = mask[neg_x_min:(neg_x_min+tmp_width),neg_y_min:(neg_y_min+tmp_height)]
                flag = np.any(tmp_rect == 0)   # overlap with positive boxes
                end_time = time.time()
                #print(end_time-start_time)
                if (end_time-start_time)>5:        # if takes more than 10 seconds,this should be stop
                    print("time takes more than 10 secs")
                    flag = False

            neg_boxes.append((neg_x_min,neg_y_min,(neg_x_min+tmp_width),(neg_y_min+tmp_height)))

    return neg_boxes


def extract_hog_feature(img_file,label_info_file,feat_save_path,with_neg = False):
    '''
    extract hog feature from one image
    attention: one image may contains several classes of object

    :param img_file:        image to be extracted
    :param label_info_file: object label information file
    :param feat_save_path: where to save the hog feature file
    :return:
    '''

    id_and_rect = read_label_info_from_file(label_info_file)
    im = cv2.imread(img_file)
    gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    print(gray.shape)
    i = 0
    basename = os.path.basename(img_file).split(".")[0]
    for (id,rect) in id_and_rect.items():
        print("id= "+ str(id))
        target_path = os.path.join(feat_save_path,str(id))
        if not os.path.exists(target_path):
            os.mkdir(target_path)
        for rec in rect:
            #print(rec[0],rec[1],rec[2],rec[3])
            roi = gray[int(rec[1]):int(rec[3]),int(rec[0]):int(rec[2])]  # caution : the array sequence
            if with_neg and int(id)<3:
                neg_target_path = os.path.join(feat_save_path, str(id) + "_neg")
                if not os.path.exists(neg_target_path):
                    os.mkdir(neg_target_path)
                neg_boxes = generate_neg_box(gray.shape[0],gray.shape[1],rect)
                for neg_box in neg_boxes:
                    (xmin, ymin, xmax, ymax) = neg_box
                    neg_roi = gray[ymin:ymax,xmin:xmax]   # cautious sequence of x and y
                    # print(xmin, ymin, xmax, ymax)
                    # cv2.imshow("neg_roi",neg_roi)
                    # cv2.waitKey(0)
                    neg_roi = cv2.resize(neg_roi, (200, 250))
                    feat = feature.hog(neg_roi, orientations=9, pixels_per_cell=(10, 10),
                                       cells_per_block=(2, 2), transform_sqrt=True)
                    joblib.dump(feat, os.path.join(neg_target_path, basename + str(i) + ".feat"))
                    cv2.imwrite(os.path.join(neg_target_path, basename + str(i) + ".jpg"),neg_roi)
                    i = i + 1
            # for neg_box in neg_boxes:
            #     (xmin,ymin,xmax,ymax) = neg_box
            #     cv2.rectangle(gray,(xmin,ymin),(xmax,ymax),(0,255,0))
            cv2.rectangle(gray,(int(rec[0]),int(rec[1])),(int(rec[2]),int(rec[3])),(255,255,120))
            cv2.imshow("samples ",gray)
            cv2.imshow("roi",roi)
            cv2.waitKey(0)
            roi = cv2.resize(roi,(200,250))
            print("lable is:\t",id)
            #feat = get_hog(roi)
            feat = feature.hog(roi, orientations=9, pixels_per_cell=(10, 10),
                        cells_per_block=(2, 2), transform_sqrt=True)
            joblib.dump(feat, os.path.join(target_path,basename+str(i)+".feat"))
            i = i+1

        # cv2.imshow("samples ",gray)
        # #cv2.imshow("roi",roi)
        # cv2.waitKey(0)

def get_hog(image):
    winSize = (64,64)
    image = cv2.resize(image,(200,250))
    #winSize = (image.shape[1], image.shape[0])
    blockSize = (8,8)
    # blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    winStride = (8,8)
    padding = (8,8)
    locations = [] # (10, 10)# ((10,20),)
    hist = hog.compute(image,winStride,padding,locations)
    return hist

if __name__ == '__main__':
    # img_dir = r"D:\data\imgs\images"
    # feat_dir = r"D:\data\imgs\hog_feats"
    img_dir = r"D:\data\test_carmera\svm\images"
    feat_dir = r"D:\data\test_carmera\svm\hog_feats"
    for file in os.listdir(img_dir):
        image = os.path.join(img_dir,file)
        txt_file = image.replace("images","labels").replace(".jpg",".xml.txt")
        extract_hog_feature(image,txt_file,feat_dir,with_neg =True)
        print("=======    extract hog feature from file %s  done..======="%image)
    print("====all file have extract done...")


```

参考:

http://blog.csdn.net/yjl9122/article/details/72765959

https://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/

https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/





