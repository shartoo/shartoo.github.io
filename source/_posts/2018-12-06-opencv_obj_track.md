---
layout: post
title: 图像处理：opencv的目标追踪方法总结
description: 图像处理
category: 图像处理
---

参考自： https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/ 和 https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/

##  什么是目标追踪

在视频后续帧中定位一个物体，称为追踪。虽然定义简单，但是目标追踪是一个相对广义的定义，比如以下问题 也属于目标追踪问题：
1. **稠密光流**：此类算法用来评估一个视频帧中的**每个像素的运动向量**
2. **稀疏光流**：此类算法，像Kanade-Lucas-Tomashi(KLT)特征追踪，追踪一张图片中**几个特征点**的位置
3. **Kalman Filtering**：一个非常出名的**信号处理算法**基于先前的运动信息用来预测运动目标的位置。早期用于导弹的导航
4. **MeanShift和Camshift**：这些算法是用来**定位密度函数的最大值**，也用于追踪
5. **单一目标追踪**：此类追踪器中，第一帧中的用矩形标识目标的位置。然后在接下来的帧中用追踪算法。日常生活中，此类追踪器用于与目标检测混合使用。
6. **多目标追踪查找算法**：如果我们有一个非常快的目标检测器，在每一帧中检测多个目标，然后运行一个追踪查找算法，来识别当前帧中某个矩形对应下一帧中的某个矩形。

## 追踪 VS 检测

如果你使用过opencv 的人脸检测算法，你就知道算法可以实时，并且很准确的检测到每一帧中的人脸。那么，为什么首先要追踪呢？我们首先考虑几个问题：

1. **跟踪比检测更快**：通常**跟踪算法比检测算法更快**。原因很简单，当你跟踪前一帧中的某个物体时，你已经知道了此物体的外观信息。同时你也知道前一帧的位置，以及运行的速度和方向。因而，在下一帧你可以用所有的信息来预测下一帧中物体的位置，以及在一个很小范围内搜索即可得到目标的位置。好的追踪算法会利用所有已知信息来追踪点，但是检测算法每次都要重头开始。
所以，通常，如果我们在第n帧开始检测，那么我们需要在第n-1帧开始跟踪。那么为什么不简单地第一帧开始检测，并从后续所有帧开始跟踪。因为跟踪会利用其已知信息，但是也可能会丢失目标，因为目标可能被障碍物遮挡，甚至于目标移动速度太快，算法跟不上。通常，跟踪算法会累计误差，而且bbox 会慢慢偏离目标。为了修复这些问题，需要不断运行检测算法。检测算法用大量样本训练之后，更清楚目标类别的大体特征。另一方面，跟踪算法更清楚它所跟踪的类别中某一个特定实例。

2. **检测失败的话，跟踪可以帮忙**：如果你在视频中检测人脸，然后人脸被某个物体遮挡了，人脸检测算法大概率会失败。好的跟踪算法，另一方面可以处理一定程度的遮挡

3. **跟踪会保存实体**：目标检测算法的输出是包含物体的一个矩形的数组，。但是没有此物体的个体信息，


## Opencv 3 的跟踪API

opencv实现了7中跟踪算法，但是3.4.1以及以上版本才有完整的7种。`BOOSTING`, `MIL`, `KCF`, `TLD`, `MEDIANFLOW`, `GOTURN`, `MOSSE`,`CSRT`。


## 代码

C++代码

```
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
 
using namespace cv;
using namespace std;
 
// Convert to string
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()
 
int main(int argc, char **argv)
{
    // List of tracker types in OpenCV 3.4.1
    string trackerTypes[8] = {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"};
    // vector <string> trackerTypes(types, std::end(types));
 
    // Create a tracker
    string trackerType = trackerTypes[2];
 
    Ptr<Tracker> tracker;
 
    #if (CV_MINOR_VERSION < 3)
    {
        tracker = Tracker::create(trackerType);
    }
    #else
    {
        if (trackerType == "BOOSTING")
            tracker = TrackerBoosting::create();
        if (trackerType == "MIL")
            tracker = TrackerMIL::create();
        if (trackerType == "KCF")
            tracker = TrackerKCF::create();
        if (trackerType == "TLD")
            tracker = TrackerTLD::create();
        if (trackerType == "MEDIANFLOW")
            tracker = TrackerMedianFlow::create();
        if (trackerType == "GOTURN")
            tracker = TrackerGOTURN::create();
        if (trackerType == "MOSSE")
            tracker = TrackerMOSSE::create();
        if (trackerType == "CSRT")
            tracker = TrackerCSRT::create();
    }
    #endif
    // Read video
    VideoCapture video("videos/chaplin.mp4");
     
    // Exit if video is not opened
    if(!video.isOpened())
    {
        cout << "Could not read video file" << endl; 
        return 1; 
    } 
 
    // Read first frame 
    Mat frame; 
    bool ok = video.read(frame); 
 
    // Define initial bounding box 
    Rect2d bbox(287, 23, 86, 320); 
 
    // Uncomment the line below to select a different bounding box 
    // bbox = selectROI(frame, false); 
    // Display bounding box. 
    rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 ); 
 
    imshow("Tracking", frame); 
    tracker->init(frame, bbox);
     
    while(video.read(frame))
    {     
        // Start timer
        double timer = (double)getTickCount();
         
        // Update the tracking result
        bool ok = tracker->update(frame, bbox);
         
        // Calculate Frames per second (FPS)
        float fps = getTickFrequency() / ((double)getTickCount() - timer);
         
        if (ok)
        {
            // Tracking success : Draw the tracked object
            rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
        }
        else
        {
            // Tracking failure detected.
            putText(frame, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
        }
         
        // Display tracker type on frame
        putText(frame, trackerType + " Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
         
        // Display FPS on frame
        putText(frame, "FPS : " + SSTR(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);
 
        // Display frame.
        imshow("Tracking", frame);
         
        // Exit if ESC pressed.
        int k = waitKey(1);
        if(k == 27)
        {
            break;
        }
 
    }
}
```

Python代码

```
import cv2
import sys
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')￼
if __name__ == '__main__' :
    # Set up tracker.
    # Instead of MIL, you can also use
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[2]
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
    # Read video
    video = cv2.VideoCapture("videos/chaplin.mp4")
    # Exit if video not opened.
    if not video.isOpened():
        print "Could not open video"
        sys.exit()
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print 'Cannot read video file'
        sys.exit()
     
    # Define an initial bounding box
    bbox = (287, 23, 86, 320)
    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
        # Start timer
        timer = cv2.getTickCount()
        # Update tracker
        ok, bbox = tracker.update(frame)
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
        # Display result
        cv2.imshow("Tracking", frame)
 
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
```

## 追踪算法详细

首先，思考下跟踪时我们的目标是在当前帧找到在前面的所有或绝大部分帧中正确跟踪的目标。由于我们追踪目标到当前帧，所以我们已知它是如何运动的，也即我们知道运行模型的参数。所谓运行模型，就是我们知道前面帧中的目标的位置和速度。即便你对目标一无所知，但是根据当前运行模型也可以预测目标的可能位置，并且这个预测可能会很准确。

但是，我们有更多的信息，比如我们可以对目标进行编码来代表目标的外观，这样的外观模型。此模型可以用来搜索，由运动模型预测的临近范围内的目标来获得更加准确的预测。

***运动模型预测目标的大概位置，外观模型微调此预估，来获得一个更准确的预测***

如果说目标很简单，并且外观改变不大的话，可以简单地使用一个模板作为外观模型并在图像中搜索模板就可以。
分类器的任务就是简单地判别一个矩形框是目标还是背景。分类器的输入是一个图像patch，返回一个0到1之间的分值。分值为0代表是背景，1则为目标。
在机器学习中，我们通常用**在线学习**代表算法可以在运行时飞快地训练完，而离线分类器需要几千个样本来训练一个分类器，而在线算法仅需要几个样本就可以。
    
分类器由正样本(目标)和负样本(非目标)来训练得到，以此分类器学习到目标与非目标之间的差异。在训练在线分类器时，我们没有数量众多的正负样本。

### Boost 追踪器

此跟踪器基于**在线版本的AdaBoost**，这个是以Haar特征级联的人脸检测器内部使用。此分类器需要在运行时以正负样本来训练。

1. 其初始框由用户指定，作为追踪的正样本，而在框范围之外许多其他patch都作为背景。
2. 在新的一帧图像中，分类器在前一帧框的周围的每个像素上分类，并给出得分。
3. 目标的新位置即得分最高的
4. 这样一来有新的正样本来重新训练分类器。依次类推。

**优点**：几乎没有，几十年前的技术。
**缺点**：追踪性能一般，它无法感知追踪什么时候会失败。

### MIL追踪

算法与Boost很像，唯一的区别是，它会考虑当前标定框周围小部分框同时作为正样本，你可能认为这个想法比较烂，因为大部分的这些`正样本`其实目标并不在中心。

这就是MIL(Multiple Instance Learning)的独特之处，在MIL中你不需要指定正负样本，而是**正负样包(bags)**。在正样本包中的并不全是正样本，而是仅需要一个样本是正样本即可。当前示例中，正样本包里面的样本包含的是处于中心位置的框，以及中心位置周围的像素所形成的框。即便当前位置的跟踪目标不准确，从以当前位置为中心在周围像素抽取的样本框所构成的正样本包中，仍然有很大概率命中一个恰好处于中心位置的框。

**优点**：性能很好，不会像Boost那样会偏移，即便出现部分遮挡依然表现不错。 **在Opencv3.0中效果最好的追踪器，如果在更高版本里，选择KCF**
**缺点**：没法应对全遮挡，追踪失败也较难的得到反馈。

### KCF 追踪

KCF即Kernelized Correlation Filters,思路借鉴了前面两个。注意到MIL所使用的多个正样本之间存在交大的重叠区域。这些重叠数据可以引出一些较好的数学特性，这些特性同时可以用来构造更快更准确的分类器。

**优点**：速度和精度都比MIL效果好，并且追踪失败反馈比Boost和MIL好。Opencv 3.1以上版本最好的分类器。
**缺点**：完全遮挡之后没法恢复。

### TLD追踪

TLD即Tracking, learning and detection，如其名此算法由三部分组成`追踪`,`学习`,`检测`。追踪器逐帧追踪目标，检测器定位所有到当前为止观察到的外观，如果有必要则纠正追踪器。`学习`会评估检测器的错误并更新，以避免进一步出错。
此追踪器可能会产生跳跃，比如你正在追踪一个人，但是场景中存在多个人，此追踪器可能会突然跳到另外一个行人进行追踪。
优势是：此追踪器可以应对大尺度变换，运行以及遮挡问题。如果说你的视频序列中，某个物体隐藏在另外一个物体之后，此追踪器可能是个好选择。 

**优势**：多帧中被遮挡依然可以被检测到，目标尺度变换也可以被处理。
**缺点**：容易误跟踪导致基本不可用。

### MedianFlow 追踪

此追踪器在视频的前向时间和后向时间同时追踪目标，然后评估两个方向的轨迹的误差。最小化前后向误差，使得其可以有效地检测追踪失败的清情形，并在视频中选择相对可靠的轨迹。 

实测时发现，**此追踪器在运动可预测以及运动速度较小时性能最好。而不像其他追踪器那样，即便追踪失败继续追踪，此追踪器很快就知道追踪失败**

**优点**：对追踪失败反馈敏锐，当场景中运动可预测以及没有遮挡时，较好
**缺点**：急速运动场景下会失败。

### GoTurn 追踪

这个是唯一使用CNN方法的追踪器，此算法对视点变化，光照变换以及形变等问题的鲁棒性较好。但是无法处理遮挡问题。

**注意**：它使用caffe模型来追踪，需要下载caffe模型以及proto txt文件。

参考[这篇文章](https://www.learnopencv.com/goturn-deep-learning-based-object-tracking/)详细介绍如何使用 GoTurn

### MOSSE 追踪

MOSSE即Minimum Output Sum of Squared Error，使用一个自适应协相关来追踪，产生稳定的协相关过滤器，并使用单帧来初始化。MOSSE鲁棒性较好，可以应对光线变换，尺度变换，姿势变动以及非网格化的变形。它也能基于峰值与旁瓣比例来检测遮挡，这使得追踪可以在目标消失时暂停并在目标出现时重启。MOSSE可以在高FPS(高达450以上)的场景下运行。易于实现，并且与其他复杂追踪器一样精确，并且可以更快。但是，在性能尺度上看，大幅度落后于基于深度学习的追踪器。


### CSRT 追踪

在 Discriminative Correlation Filter with Channel and Spatial Reliability （DCF-CSR）中，我们使用空间依赖图来调整过滤器支持

## 总结

+ 如果需要更高的准确率，并且可以容忍延迟的话，使用CSRT
+ 如果需要更快的FPS，并且可以容许稍低一点的准确率的话，使用KCF
+ 如果纯粹的需要速度的话，用MOSSE


