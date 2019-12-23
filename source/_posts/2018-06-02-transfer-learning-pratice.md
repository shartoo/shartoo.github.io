---
layout: post
title: 迁移学习实践-Tensorflow分类任务
description: 深度学习基础理论
category: blog
---

摘自 : [medium transfer learning](https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a)

## 1 说明和准备

### 1.1 任务问题

我们要对只有4000张图片(3000张训练，1000张验证)的数据集做图像分类，分为`猫`和`狗`两类。图片数据可以从[kaggle 猫狗分类挑战](https://www.kaggle.com/c/dogs-vs-cats/data)上下载到25000张，不过为了演示迁移学习，假定只有4000张图片。

### 1.2 数据准备

首先下载全部的数据集，然后筛选出其中的4000张。

```
import glob
import numpy as np
import os
import shutil
np.random.seed(42)
files = glob.glob('train/*')
# 载入全部的猫狗图片
cat_files = [fn for fn in files if 'cat' in fn]
dog_files = [fn for fn in files if 'dog' in fn]
print(len(cat_files), len(dog_files))
# (12500, 12500)

# 筛选出其中的4000张图片
cat_train = np.random.choice(cat_files, size=1500, replace=False)
dog_train = np.random.choice(dog_files, size=1500, replace=False)
cat_files = list(set(cat_files) - set(cat_train))
dog_files = list(set(dog_files) - set(dog_train))

cat_val = np.random.choice(cat_files, size=500, replace=False)
dog_val = np.random.choice(dog_files, size=500, replace=False)
cat_files = list(set(cat_files) - set(cat_val))
dog_files = list(set(dog_files) - set(dog_val))

cat_test = np.random.choice(cat_files, size=500, replace=False)
dog_test = np.random.choice(dog_files, size=500, replace=False)

print('Cat datasets:', cat_train.shape, cat_val.shape, cat_test.shape)
print('Dog datasets:', dog_train.shape, dog_val.shape, dog_test.shape)
# Cat datasets: (1500,) (500,) (500,)
# Dog datasets: (1500,) (500,) (500,)
```
将数据子集单独放到其他文件夹

```
train_dir = 'training_data'
val_dir = 'validation_data'
test_dir = 'test_data'

train_files = np.concatenate([cat_train, dog_train])
validate_files = np.concatenate([cat_val, dog_val])
test_files = np.concatenate([cat_test, dog_test])

os.mkdir(train_dir) if not os.path.isdir(train_dir) else None
os.mkdir(val_dir) if not os.path.isdir(val_dir) else None
os.mkdir(test_dir) if not os.path.isdir(test_dir) else None

for fn in train_files:
    shutil.copy(fn, train_dir)

for fn in validate_files:
    shutil.copy(fn, val_dir)
    
for fn in test_files:
    shutil.copy(fn, test_dir)
```

为模型准备数据

```
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

IMG_DIM = (150, 150)

train_files = glob.glob('training_data/*')
train_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in train_files]
train_imgs = np.array(train_imgs)
train_labels = [fn.split('\\')[1].split('.')[0].strip() for fn in train_files]

validation_files = glob.glob('validation_data/*')
validation_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in validation_files]
validation_imgs = np.array(validation_imgs)
validation_labels = [fn.split('\\')[1].split('.')[0].strip() for fn in validation_files]

print('Train dataset shape:', train_imgs.shape, 
      '\tValidation dataset shape:', validation_imgs.shape)
# Train dataset shape: (3000, 150, 150, 3)  
# Validation dataset shape: (1000, 150, 150, 3)
```
现在，我们得到了3000张训练集和1000张验证集，图像长宽为$150\times 150$，接下来，我们要将图片像素矩阵值取值范围缩放到0到1.

```
train_imgs_scaled = train_imgs.astype('float32')
validation_imgs_scaled = validation_imgs.astype('float32')
train_imgs_scaled /= 255
validation_imgs_scaled /= 255

print(train_imgs[0].shape)
array_to_img(train_imgs[0])
```

一些基本参数，同时将字符型的分类类别改为数值型。

```
input_shape = (150, 150, 3)

# encode text category labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(train_labels)
train_labels_enc = le.transform(train_labels)
validation_labels_enc = le.transform(validation_labels)

print(train_labels[1495:1505], train_labels_enc[1495:1505])
# ['cat', 'cat', 'cat', 'cat', 'cat', 'dog', 'dog', 'dog', 'dog', 'dog'] [0 0 0 0 0 1 1 1 1 1]
```

## 2 基准模型

先手写个基准的CNN模型，如下

```
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras import optimizers

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', 
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(),
              metrics=['accuracy'])

model.summary()
```
模型架构如下

```
Layer (type) Output Shape Param #   
=================================================================
conv2d_1 (Conv2D) (None, 148, 148, 16) 448       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 74, 74, 16) 0         
_________________________________________________________________
conv2d_2 (Conv2D) (None, 72, 72, 64) 9280      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 36, 36, 64) 0         
_________________________________________________________________
conv2d_3 (Conv2D) (None, 34, 34, 128) 73856     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 17, 17, 128) 0         
_________________________________________________________________
flatten_1 (Flatten) (None, 36992) 0         
_________________________________________________________________
dense_1 (Dense) (None, 512) 18940416  
_________________________________________________________________
dense_2 (Dense) (None, 1) 513       
=================================================================
Total params: 19,024,513
Trainable params: 19,024,513
Non-trainable params: 0
```

我们设置`batch_size=30`，总共有3000张图片，也就是一个epoch需要100次迭代。我们训练30个epoch，然后验证模型。

```
batch_size = 30
num_classes = 2
epochs = 30
history = model.fit(x=train_imgs_scaled, y=train_labels_enc,
                    validation_data=(validation_imgs_scaled, validation_labels_enc),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)
```
训练过程的输出如下:

```
Train on 3000 samples, validate on 1000 samples
Epoch 1/30
3000/3000 - 10s - loss: 0.7583 - acc: 0.5627 - val_loss: 0.7182 - val_acc: 0.5520
Epoch 2/30
3000/3000 - 8s - loss: 0.6343 - acc: 0.6533 - val_loss: 0.5891 - val_acc: 0.7190
...
...
Epoch 29/30
3000/3000 - 8s - loss: 0.0314 - acc: 0.9950 - val_loss: 2.7014 - val_acc: 0.7140
Epoch 30/30
3000/3000 - 8s - loss: 0.0147 - acc: 0.9967 - val_loss: 2.4963 - val_acc: 0.7220
```
训练集都接近100%准确率了，但是验证集准确率还只有72%。模型可能存在过拟合。可以使用如下的代码画出训练和验证过程的loss下降和准确率变化。

```
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Basic CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(1,31))
ax1.plot(epoch_list, history.history['acc'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_acc'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, 31, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, 31, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")
```
![](/images/blog/transfer_learning_pratice_1.png)

上图左可以看到，3个epoch之后就开始过拟合了，训练准确率一直上升，但是验证准确率保持不变了。

### 2.1 简单的优化模型

上面的CNN是个基本的架构，接下来我们做一些优化策略，网络架构上加入正则化，使用一定概率的dropout。只修改网络结构部分，其他训练代码不变。

```
model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', 
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(),
              metrics=['accuracy'])          
history = model.fit(x=train_imgs_scaled, y=train_labels_enc,
                    validation_data=(validation_imgs_scaled, validation_labels_enc),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)                      
```
使用上面的画图代码，画出训练曲线。

![](/images/blog/transfer_learning_pratice_2.png)

有所改善，但是效果不明显。依然是过拟合，究其原因，数据量太少，可以使用部分的数据集增强策略增加数据多样性。

### 2.2 使用数据增强策略

keras的`ImageDataGenerator`自带了一些数据增强方法，如下

```
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 
                                   horizontal_flip=True, fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1./255)
```
当然，我们还可以使用 [Albumentations](https://github.com/albu/albumentations)来做更多的增强策略。我们先来看看`ImageDataGenerator`增强之后的图片效果

```
mg_id = 2595
cat_generator = train_datagen.flow(train_imgs[img_id:img_id+1], train_labels[img_id:img_id+1],
                                   batch_size=1)
cat = [next(cat_generator) for i in range(0,5)]
fig, ax = plt.subplots(1,5, figsize=(16, 6))
print('Labels:', [item[1][0] for item in cat])
l = [ax[i].imshow(cat[i][0][0]) for i in range(0,5)]
```

![](/images/blog/transfer_learning_pratice_3.png)

再次使用上面的基准模型(加了dropout层的)，此次我们将学习率稍微改小点，将默认的`1e-3`改为`1e-4`，防止模型过拟合，此时的数据量增多了。

```
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])        
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100,
                              validation_data=val_generator, validation_steps=50, 
                              verbose=1)  
```
打印训练曲线
![](/images/blog/transfer_learning_pratice_4.png)

模型的准确率提升到**82%**，而且已经不再过拟合了。

## 3 使用其他模型做迁移学习

### 3.1 VGG-16模型

分类模型，我们选用VGG16为例。首先，我们需要理解VGG16的模型架构，如下

![](/images/blog/transfer_learning_pratice_5.png)

13个$3\times 3$的卷积，5个maxpooling缩减了网络输入尺寸。在两个全连接层之前的输出是4096个神经元，全连接都是1000个神经元(代表了1000个分类)。由于我们要做的是做猫狗分类，最后三层是不需要的。我们更关心前5个blocks(下图)，我们可以将VGG模型看做一个特征抽取器。下图是VGG模型的三种用法

![](/images/blog/transfer_learning_pratice_6.png)

+ 如果我们只是作为特征抽取器，则按照图中中间的示例，冻结所有的blocks(5个)，在训练过程中，这些blocks中的所有参数都不会更新。
+ 如果我们做fine-tuning，可以考虑按照图右冻结前3个blocks，更新后面两个blocks(4和5)的参数（每个训练epoch过程都会）。

### 3.2 将预训练模型作为特征抽取器

3.1节中最后一张图的中间的架构，冻结所有blocks层的参数的用法。下面是对应的代码

```
from keras.applications import vgg16
from keras.models import Model
import keras

vgg = vgg16.VGG16(include_top=False, weights='imagenet', 
                                     input_shape=input_shape)

output = vgg.layers[-1].output
output = keras.layers.Flatten()(output)
vgg_model = Model(vgg.input, output)

vgg_model.trainable = False
for layer in vgg_model.layers:
    layer.trainable = False
    
import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]
pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable']) 
```
![](/images/blog/transfer_learning_pratice_7.png)

此处，将VGG模型看做SURF或者HOG特征之类的东西就可以，使用过程不更新参数，直接输入图片，预测得到特征。用法如下

```
bottleneck_feature_example = vgg.predict(train_imgs_scaled[0:1])
print(bottleneck_feature_example.shape)
plt.imshow(bottleneck_feature_example[0][:,:,0])
```
![](/images/blog/transfer_learning_pratice_8.png)

从训练数据和验证数据中使用VGG16抽取特征如下

```
def get_bottleneck_features(model, input_imgs):
    features = model.predict(input_imgs, verbose=0)
    return features

train_features_vgg = get_bottleneck_features(vgg_model, train_imgs_scaled)
validation_features_vgg = get_bottleneck_features(vgg_model, validation_imgs_scaled)
print('Train Bottleneck Features:', train_features_vgg.shape, 
      '\tValidation Bottleneck Features:', validation_features_vgg.shape)
```
输出如下

```
Train Bottleneck Features: (3000, 8192)  
Validation Bottleneck Features: (1000, 8192)
```
接下来，我们可以以VGG作为特征抽取器重新构建一个训练模型。其实，在抽取特征之后直接接一个SVM或者KNN分类器也是一样的。下面以keras代码，重新构建CNN模型

```
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers
# 此时的vgg_model 已经设置了trainable=False
input_shape = vgg_model.output_shape[1]

model = Sequential()
model.add(InputLayer(input_shape=(input_shape,)))
model.add(Dense(512, activation='relu', input_dim=input_shape))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])

model.summary()
```
网络结构如下

```
_________________________________________________________________
Layer (type) Output Shape Param #   
=================================================================
input_2 (InputLayer) (None, 8192) 0         
_________________________________________________________________
dense_1 (Dense) (None, 512) 4194816   
_________________________________________________________________
dropout_1 (Dropout) (None, 512) 0         
_________________________________________________________________
dense_2 (Dense) (None, 512) 262656    
_________________________________________________________________
dropout_2 (Dropout) (None, 512) 0         
_________________________________________________________________
dense_3 (Dense) (None, 1) 513       
=================================================================
Total params: 4,457,985
Trainable params: 4,457,985
Non-trainable params: 0
```
**注意，此时的训练代码中输入数据不再是图片，而是VGG抽取的特征了**

```
history = model.fit(x=train_features_vgg, y=train_labels_enc,
                    validation_data=(validation_features_vgg, validation_labels_enc),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)
```

![](/images/blog/transfer_learning_pratice_9.png)

验证准确率提升到**88%**，虽然看起来依然过拟合了。

### 3.3 使用数据增强+VGG作为特征抽取器

由于我们使用data generator，此处不再用VGG作为特征抽取器.此部分与上面的区别在于，上面的VGG模型不是网络的一部分，属于数据处理部分，用vgg将处理图片(特征抽取)之后的数据传入了新的小网络。而当前是将VGG作为网络的一部分，与新的网络层，构建了一个新模型

```
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 
                                   horizontal_flip=True, fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow(train_imgs, train_labels_enc, batch_size=30)
val_generator = val_datagen.flow(validation_imgs, validation_labels_enc, batch_size=20)
```
网络构建部分
```
from keras.applications import vgg16
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers
from keras.models import Model
import keras

vgg = vgg16.VGG16(include_top=False, weights='imagenet', 
                                     input_shape=input_shape)

output = vgg.layers[-1].output
output = keras.layers.Flatten()(output)
vgg_model = Model(vgg.input, output)

vgg_model.trainable = False
for layer in vgg_model.layers:
    layer.trainable = False
# 下面是我们新加的网络层，将VGG放在前面
model = Sequential()
model.add(vgg_model)
model.add(Dense(512, activation='relu', input_dim=input_shape))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
# 学习率变小了
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
              
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100,
                              validation_data=val_generator, validation_steps=50, 
                              verbose=1)       
```
此时的学习曲线，如下

![](/images/blog/transfer_learning_pratice_10.png)

此时的验证准确率提升到了90%，而且没有过拟合。

### 3.5 fine-tuning 预训练的VGG模型+数据增强

此部分，可以参考3.1节部分VGG示意图的最右边那张图，此时vgg模型中某些blocks的参数也在训练过程中得到更新。如下

```
vgg_model.trainable = True

set_trainable = False
for layer in vgg_model.layers:
    if layer.name in ['block5_conv1', 'block4_conv1']:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
        
layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]
pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])   
```
![](/images/blog/transfer_learning_pratice_11.png)

可以看到`block4`和`block5`已经变成可以训练了。此时再次减小学习率，同时使用了增强了的数据处理

```
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 
                                   horizontal_flip=True, fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow(train_imgs, train_labels_enc, batch_size=30)
val_generator = val_datagen.flow(validation_imgs, validation_labels_enc, batch_size=20)

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers

model = Sequential()
model.add(vgg_model)
model.add(Dense(512, activation='relu', input_dim=input_shape))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['accuracy'])
              
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100,
                              validation_data=val_generator, validation_steps=50, 
                              verbose=1)              
```

![](/images/blog/transfer_learning_pratice_12.png)

可以看到，验证准确率已经提升到了**96%**，与基准模型相比，提升了24%。

## 4 测试模型

接下来在测试集上测试上面的5种模型

1. 基准CNN模型
2. 使用了数据增强的基准CNN模型
3. 迁移学习：使用VGG16作为特征抽取器【VGG只用在数据处理上】
4. 迁移学习：使用VGG作为模型的一部分，并且使用了数据增强策略
5. 迁移学习：对VGG模型微调，让其`block4`和`block5`参数可更新

测试代码

```
IMG_DIM = (150, 150)

test_files = glob.glob('test_data/*')
test_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in test_files]
test_imgs = np.array(test_imgs)
test_labels = [fn.split('/')[1].split('.')[0].strip() for fn in test_files]

test_imgs_scaled = test_imgs.astype('float32')
test_imgs_scaled /= 255
test_labels_enc = class2num_label_transformer(test_labels)

print('Test dataset shape:', test_imgs.shape)
print(test_labels[0:5], test_labels_enc[0:5])
```
测试基准模型的代码如下（其他模型类似）

```
predictions = basic_cnn.predict_classes(test_imgs_scaled, verbose=0)
predictions = num2class_label_transformer(predictions)
meu.display_model_performance_metrics(true_labels=test_labels, predicted_labels=predictions, 
                                      classes=list(set(test_labels)))
```

![](/images/blog/transfer_learning_pratice_13.png)
