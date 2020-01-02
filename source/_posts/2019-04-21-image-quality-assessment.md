---
layout: post
title: 图像质量评估
description: 图像处理
category: 图像处理
mathjax: true
---

参考 [Image Quality Assessment](https://github.com/ocampor/notebooks/blob/master/notebooks/image/quality/brisque.ipynb?source=post_page-----391a6be52c11----------------------)

## 1 说明

图像质量评估(Image Quality Assessment(IQA))看起来是个非常主观的事，但是可以借鉴一些方法，目前主流方法分两类

1. 基于引用比较的评估。
2. 无引用的评估

主要区别是，基于引用的评估方法需要依赖一张高质量的图片作为评估源，来比较两张图之间的区别。常用的基于引用的评估方法是结构相似性索引(Structural Similarity Index(SSIM))。

## 2 无引用图片质量评估

它不需要引用图片，仅仅依赖于接收到的图片信息，称为盲测方法。分为两步(1)计算能够描述图片结构的特征(2)计算与人类对于图片质量观点相关的特征。TID2008是一个基于方法论的数据集，它描述了如何从引用图片中评估人类的观点，被广泛用于对比IQA算法的性能。

### 2.1 Blind/referenceless image spatial quality evaluator (BRISQUE)

BRISQUE是一种仅使用图像像素来计算特征(其他方法都是基于图像转换到其他空间，比如wavelet 或者DCT)。非常高效，因为它不需要其他任何信息来计算其特征。

它依赖于空间域中局部正规化的亮度系数的空间自然场景统计(Spatial Natural Scene Statistics(NSS))模型，以及这些系数的点与点之间的内积的模型。

### 2.2 方法论

#### 2.2.1 Natural Scene Statistics in the Spatial Domain

给定图像$I(i,j)$，首先通过减去局部均值$\mu (i,j)$，然后除以局部方差$\delta(i,j)$ 来计算局部亮度系数$\hat I(i,j)$。加上$C$是为了避免除以0.

$$
\hat I(i,j) = \frac{I(i,j)-\mu(i,j)}{\delta(i,j)+c}\\
其中如果I(i,j)\in [0,255]，则C=1,如果 \in [0,1]，那么C=1/255
$$
为了计算局部归一化的亮度，即平均减去的对比度归一化(MSCN)系数，首先，我们需要计算局部均值。
$$
\mu (i,j) = \sum _{k=-K} ^{K} \sum _{I=-k} ^L w_{k,l}I_{k,l}(i,j) \\
其中w是尺寸为(K,L)的高斯核
$$
计算代码

```
def normalize_kernel(kernel):
    return kernel / np.sum(kernel)

def gaussian_kernel2d(n, sigma):
    Y, X = np.indices((n, n)) - int(n/2)
    gaussian_kernel = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2)) 
    return normalize_kernel(gaussian_kernel)

def local_mean(image, kernel):
    return signal.convolve2d(image, kernel, 'same')
```
然后计算局部偏差
$$
\sigma(i,j)\sqrt{\sum_{k=-K} ^K \sum _{l=-L} ^L w_{k,l}(I_{k,l}(i,j)-\mu(i,j))^2 }
$$
代码

```
def local_deviation(image, local_mean, kernel):
    "Vectorized approximation of local deviation"
    sigma = image ** 2
    sigma = signal.convolve2d(sigma, kernel, 'same')
    return np.sqrt(np.abs(local_mean ** 2 - sigma)
```
最后，我们可以计算得到MSCN系数
$$
\hat I(i,j) = \frac{I(i,j)-\mu(i,j)}{\sigma(i,j)+C}
$$

```
def calculate_mscn_coefficients(image, kernel_size=6, sigma=7/6):
    C = 1/255
    kernel = gaussian_kernel2d(kernel_size, sigma=sigma)
    local_mean = signal.convolve2d(image, kernel, 'same')
    local_var = local_deviation(image, local_mean, kernel)
    
    return (image - local_mean) / (local_var + C)
```
作者发现一个扭曲的图片的MSCN系数服从一个广义高斯分布(GGD)
$$
f(x;\alpha,\sigma ^2)=\frac{\alpha}{2\beta T(1/\alpha)}e^{-(\frac{|x|}{\beta})^{\alpha}} \\
其中 \beta = \sigma\sqrt{\frac{T(\frac{1}{\alpha})}{T(\frac{3}{\alpha})}},T是伽马函数，\alpha的形状控制形状以及\sigma ^2的方差
$$

```
def generalized_gaussian_dist(x, alpha, sigma):
    beta = sigma * np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))
    coefficient = alpha / (2 * beta() * special.gamma(1 / alpha))
    return coefficient * np.exp(-(np.abs(x) / beta) ** alpha)
```

#### 2.2.2 相邻MSCN系数的点对内积

邻接的系数的符号也代表了某种特定结构，可能是某种扭曲的分布。邻接MSCN系数点对内积的沿着四个方向
1. 水平方向
2. 垂直方向
3. 主对角(main-diagonal)D1
4. 次对角(secondary-diagonal)D2

$$
D2(i,j) = \hat I(i,j)\hat I(i+1,j+1)
$$

```
def calculate_pair_product_coefficients(mscn_coefficients):
    return collections.OrderedDict({
        'mscn': mscn_coefficients,
        'horizontal': mscn_coefficients[:, :-1] * mscn_coefficients[:, 1:],
        'vertical': mscn_coefficients[:-1, :] * mscn_coefficients[1:, :],
        'main_diagonal': mscn_coefficients[:-1, :-1] * mscn_coefficients[1:, 1:],
        'secondary_diagonal': mscn_coefficients[1:, :-1] * mscn_coefficients[:-1, 1:]
    })
```
广义高斯分布不能很好的拟合系数内积的经验直方图。因此，又提出了 Asymmetric Generalized Gaussian Distribution (AGGD)[非对称广义高斯分布模型](Multiscale skewed heavy-tailed model for texture analysis. Proceedings - International Conference on Image Processing)

$$
f(x;v,\sigma _l ^2,\sigma _r ^2) = \frac{v}{(\beta _l+\beta _r)T(\frac{1}{v})}e^{(-(\frac{-x}{\beta _l})^v)} \quad\quad x<0 \\
f(x;v,\sigma _l ^2,\sigma _r ^2) = \frac{v}{(\beta _l+\beta _r)T(\frac{1}{v})}e^{(-(\frac{-x}{\beta _r})^v)} \quad\quad x\gt 0 \\
其中 \beta _{side} = \sigma _{side}\sqrt{\frac{T(\frac{1}{v})}{T(\frac{3}{v})}} ,side 可以是 r 或 l \\
前面没有提到的参数是均值 m = (\beta _r-\beta_l)\frac{T(\frac{2}{v})}{T(frac{1}{v})}
$$

```
def asymmetric_generalized_gaussian(x, nu, sigma_l, sigma_r):
    def beta(sigma):
        return sigma * np.sqrt(special.gamma(1 / nu) / special.gamma(3 / nu))
    
    coefficient = nu / ((beta(sigma_l) + beta(sigma_r)) * special.gamma(1 / nu))
    f = lambda x, sigma: coefficient * np.exp(-(x / beta(sigma)) ** nu)
        
    return np.where(x < 0, f(-x, sigma_l), f(x, sigma_r))
```

#### 2.2.3 拟合AGGD

1. 计算$\hat \gamma，其中N_l$是负样本数量，而$N_r$是正样本数量.
$$
\hat \gamma = \frac{\sqrt{\frac{1}{N_l}\sum_{k=1,x_k<0} ^{N_l}x_k ^2}}{\sqrt{\frac{1}{N_r}\sum_{k=1,x_k<0} ^{N_r}x_k ^2}}
$$

2. 计算$\hat r$
$$
\hat r = \frac{(\frac{\sum|x_k|}{N_l+N_r})^2}{\frac{\sum x_k ^2}{N_l+N_r}}
$$

3. 使用$\hat \gamma ,\hat r$计算$\hat R$
$$
\hat R = \hat r\frac{(\hat \gamma ^3+1)(\hat \gamma +1)}{(\hat \gamma ^2+1)^2}
$$

4. 使用反广义高斯比率计算$\alpha$ 
$$
\rho (\alpha) =\frac{T(2/\alpha)^2}{T(1/\alpha)T(3/\alpha)}
$$

5. 评估左右scale参数
$$
\sigma _l = sqrt{\frac{1}{N_l-1}\sum _{k=l,x_k<0} ^{N_l} x_k ^2} \\
\sigma _r = sqrt{\frac{1}{N_r-1}\sum _{k=r,x_k\gt 0} ^{N_r} x_k ^2}
$$

```
def asymmetric_generalized_gaussian_fit(x):
    def estimate_phi(alpha):
        numerator = special.gamma(2 / alpha) ** 2
        denominator = special.gamma(1 / alpha) * special.gamma(3 / alpha)
        return numerator / denominator

    def estimate_r_hat(x):
        size = np.prod(x.shape)
        return (np.sum(np.abs(x)) / size) ** 2 / (np.sum(x ** 2) / size)

    def estimate_R_hat(r_hat, gamma):
        numerator = (gamma ** 3 + 1) * (gamma + 1)
        denominator = (gamma ** 2 + 1) ** 2
        return r_hat * numerator / denominator

    def mean_squares_sum(x, filter = lambda z: z == z):
        filtered_values = x[filter(x)]
        squares_sum = np.sum(filtered_values ** 2)
        return squares_sum / ((filtered_values.shape))

    def estimate_gamma(x):
        left_squares = mean_squares_sum(x, lambda z: z < 0)
        right_squares = mean_squares_sum(x, lambda z: z >= 0)

        return np.sqrt(left_squares) / np.sqrt(right_squares)

    def estimate_alpha(x):
        r_hat = estimate_r_hat(x)
        gamma = estimate_gamma(x)
        R_hat = estimate_R_hat(r_hat, gamma)

        solution = optimize.root(lambda z: estimate_phi(z) - R_hat, [0.2]).x

        return solution[0]

    def estimate_sigma(x, alpha, filter = lambda z: z < 0):
        return np.sqrt(mean_squares_sum(x, filter))
    
    def estimate_mean(alpha, sigma_l, sigma_r):
        return (sigma_r - sigma_l) * constant * (special.gamma(2 / alpha) / special.gamma(1 / alpha))
    
    alpha = estimate_alpha(x)
    sigma_l = estimate_sigma(x, alpha, lambda z: z < 0)
    sigma_r = estimate_sigma(x, alpha, lambda z: z >= 0)
    
    constant = np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))
    mean = estimate_mean(alpha, sigma_l, sigma_r)
    
    return alpha, mean, sigma_l, sigma_r
```

#### 2.2.4 计算BRISQUE特征

计算图像质量的特征即拟合MSCN系数的结果并移动shifted内积到广义高斯分布。首先，我们需要拟合MSCN系数到GDD，然后点对内积到AGGD。特征概要如下

|FeatureID|Feature Description|Computation Procedure|
|---|---|---|
|$f_1-f_2 $ |Shape and variance|Fit GGD to MSCN coefficients|
|$f_3-f_6$|Shape, mean, left variance, right variance|Fit AGGD to **H** pairwise products|
|$f_7-f_{10}$ |Shape, mean, left variance, right variance|Fit AGGD to **V** pairwise products|
|$f_{11}-f_{14}$ |Shape, mean, left variance, right variance|Fit AGGD to **D1** pairwise products|
|$f_{15}-f_{18}$ |Shape, mean, left variance, right variance|Fit AGGD to **D2** pairwise products|

```
def calculate_brisque_features(image, kernel_size=7, sigma=7/6):
    def calculate_features(coefficients_name, coefficients, accum=np.array([])):
        alpha, mean, sigma_l, sigma_r = asymmetric_generalized_gaussian_fit(coefficients)

        if coefficients_name == 'mscn':
            var = (sigma_l ** 2 + sigma_r ** 2) / 2
            return [alpha, var]
        
        return [alpha, mean, sigma_l ** 2, sigma_r ** 2]
    
    mscn_coefficients = calculate_mscn_coefficients(image, kernel_size, sigma)
    coefficients = calculate_pair_product_coefficients(mscn_coefficients)
    
    features = [calculate_features(name, coeff) for name, coeff in coefficients.items()]
    flatten_features = list(chain.from_iterable(features))
    return np.array(flatten_features)
```

### 2.3 计算图像质量

首先，我们需要两个辅助函数

```
def plot_histogram(x, label):
    n, bins = np.histogram(x.ravel(), bins=50)
    n = n / np.max(n)
    plt.plot(bins[:-1], n, label=label, marker='o')
```

1. 载入图像
2. 计算系数.计算完MSCN系数和点对的内积之后，我们可以确定其分布实际上是不同的。

```
mscn_coefficients = calculate_mscn_coefficients(gray_image, 7, 7/6)
coefficients = calculate_pair_product_coefficients(mscn_coefficients)
for name, coeff in coefficients.items():
    plot_histogram(coeff.ravel(), name)
plt.axis([-2.5, 2.5, 0, 1.05])
plt.legend()
plt.show()
```
3. 拟合系数到广义高斯分布

```
brisque_features = calculate_brisque_features(gray_image, kernel_size=7, sigma=7/6)
```

4. resize图像并计算BRISQUE特征

```
ownscaled_image = cv2.resize(gray_image, None, fx=1/2, fy=1/2, interpolation = cv2.INTER_CUBIC)
downscale_brisque_features = calculate_brisque_features(downscaled_image, kernel_size=7, sigma=7/6)

brisque_features = np.concatenate((brisque_features, downscale_brisque_features))
```
5. 缩放特征并喂入SVR.作者提供了一个与训练的SVR模型来计算质量评估。但是，为了有个好的结果，我们需要将特征缩放到[-1,1]。对于后者，我们需要用预缩放特征向量相同的参数。

```
def scale_features(features):
    with open('normalize.pickle', 'rb') as handle:
        scale_params = pickle.load(handle)
    
    min_ = np.array(scale_params['min_'])
    max_ = np.array(scale_params['max_'])
    
    return -1 + (2.0 / (max_ - min_) * (features - min_))

def calculate_image_quality_score(brisque_features):
    model = svmutil.svm_load_model('brisque_svm.txt')
    scaled_brisque_features = scale_features(brisque_features)
    
    x, idx = svmutil.gen_svm_nodearray(
        scaled_brisque_features,
        isKernel=(model.param.kernel_type == svmutil.PRECOMPUTED))
    
    nr_classifier = 1
    prob_estimates = (svmutil.c_double * nr_classifier)()
    
    return svmutil.libsvm.svm_predict_probability(model, x, prob_estimates)
calculate_image_quality_score(brisque_features)
```

## 3 结论

方法在TID2008数据集上测试，并且效果不错，即便与引用IQA方法比起来。后续可以用XGBoost,LightGBM方法来训练识别步骤来提高效率。
