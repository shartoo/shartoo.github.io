---
layout: post
title: 使用传统方法换脸算法
description: 图像处理
category: 图像处理
---

参考 [switch face with python](https://matthewearl.github.io/2015/07/28/switching-eds-with-python/)

### 1 使用dlib抽取面部关键点

![](/images/blog/cv_switch_face_1.png)

关键代码如下
```
PREDICTOR_PATH = "/home/matt/dlib-18.16/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(im):
    rects = detector(im, 1)
    
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
```

特征抽取器`predictor`传入一个矩形的人脸部分，预测内部的人脸的68个关键点坐标，即$68\times 2$个值。

### 2 使用procrustes分析进行人脸对齐

检测两张人脸的关键点之后，每个点的特定属性我们是知道的，比如第30个点代表的是鼻尖的坐标。我们接下来要做的是，如何扭曲、转换、以及缩放第一个人脸的点，使得它与目标关键点尽可能接近。这个相同的转换步骤可以用，第二个人脸图像来覆盖第一个人脸图像来实现。

数学形式的解法为，我们寻找$T,s,R$最小化下面的等式:

$$
\sum _{i=1} ^{68}|sRp_i ^T+T-q_i ^T|^2
$$

其中$R$是一个$2\times 2$的正交矩阵，$s$是个标量，$T$是一个2向量，$p_ihe q_i$是上面计算得到的68个关键点。此问题等价于求解一个**正交procrustes分析**问题。

```
def transformation_from_points(points1, points2):
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])
```

以上代码执行了如下步骤

1. 将所有输入转换为浮点型，便于后续的计算
2. 减去每个点集合的中心坐标(即减去均值)。一旦结果点集合的最优变换和扭曲解找到，中心的`c1`和`c2`可以用来求解全局解。
3. 类似的，每个点除以标准差。消除尺度影响
4. 使用SVD计算扭曲比率，需要去查看**正交Procrustes问题**的求解过程才能了解。
5. 返回完整的转换为放射变换矩阵。

结果可以用Opencv的`cv2.wrapAffine`函数来映射第二张图到第一张图。

```
def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im
```
其实就是为了让两张图的点位能对得上，将某张图进行旋转，缩放，使得两张图的人脸的关键点能处于相同的坐标位置。

![](/images/blog/cv_switch_face_2.gif)

### 3 目标图的轮廓纠正

![](/images/blog/cv_switch_face_3.png)

Non colour-corrected overlay
接下来需要解决的问题是，两张图像的不同肤色和光照差异，会导致连接处边缘的突兀。下面的方法是尝试纠正这个问题。

```
COLOUR_CORRECT_BLUR_FRAC = 0.6
LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))

def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
                              numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += 128 * (im2_blur <= 1.0)

    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
                                                im2_blur.astype(numpy.float64))
```

![](/images/blog/cv_switch_face_4.png)

此方法尝试改变第二张图像的轮廓去匹配第一张图的。做法是：**第二张图除以其高斯模糊，然后乘以第一张图的高斯模糊**。此算法源于[RGB的缩放轮廓纠正](https://en.wikipedia.org/wiki/Color_balance#Scaling_monitor_R.2C_G.2C_and_B),但是对所有图像使用了一个常量的缩放因子，每个像素有其局部缩放因子。

由此方法，两张图像的光照差异可以在某种程度上累加。例如，如果第一张图某一边在发光二第二张图有均衡的光照，那么轮廓纠正之后图二会出现出现某些暗处。也就是说这是个比较粗暴的方案，合适大小的高斯核是问题的关键。太小的话会导致图一种某些面部特征会出现在图二中，太大的话会导致核外面某些像素重叠，并出现变色。此处使用的是$0.6\times 瞳孔距离$。

### 4 将图二特征渲染回图一

使用一个mask从图二中抽取部分，渲染到图一中。

![](/images/blog/cv_switch_face_5.png)

+ 值为1的区域(上图中的白色区域)对应的是图二中人脸特征选取的部分
+ 值为0的区域(上图中黑色区域)对应的是图一该出现的部分。0到1之间的区域是两张图的混合。

下面代码是生成如上区域的

```
LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_BROW_POINTS = list(range(17, 22))
NOSE_POINTS = list(range(27, 35))
MOUTH_POINTS = list(range(48, 61))
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]
FEATHER_AMOUNT = 11

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = numpy.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im

mask = get_face_mask(im2, landmarks2)
warped_mask = warp_im(mask, M, im1.shape)
combined_mask = numpy.max([get_face_mask(im1, landmarks1), warped_mask],
                          axis=0)
```

上面代码分解如下

1. `get_face_mask()`方法是用来定义生成图像的一个mask和一个面部关键点的矩阵。它画了两个白色凸多边形:一个环绕着眼睛区域，一个环绕着鼻子和嘴巴区域。接着它会羽化mask的边缘11个像素。羽化有助于隐藏遗留的颜色不连续问题

2. 此类面部的mask会给两张图都生成。图二的mask会被转换进图一的坐标空间，使用步骤2相同的转换。
3. 此mask接下来会逐像素取最大值。结合两个mask可以保证图一中的特征会被覆盖，同时图二中的特征也会被展示。

最后，mask用于得到最终的合成结果。

```
output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
```

![](/images/blog/cv_switch_face_6.png)
