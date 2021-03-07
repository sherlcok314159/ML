CNN 一共分为输入，卷积，池化，拉直，softmax，输出

![](https://github.com/sherlcok314159/ML/blob/main/Images/cnn.png)

**为什么CNN可以做影像识别？**

***A.*** 其实很多时候，不用看整张图片，看其中一小部分就可以判断这张图是什么了。比如说，判断一张图片里面有没有人，不需要看整张图片，看到人的头或者上半身就已经可以决定这件事情了

***B.*** 需要识别的小部分通常分布在不同图片的不同部分，我们通过让不同神经元（neuron）共享权重（weights），即共用一个滤波器（filter）来发现不同图片上的相同部分，即使它们分布在不同位置。这样一来，可以很大程度上减少参数，提高训练速度。


接下来，以**影像识别**为例具体介绍一下CNN

比如我们要识别手写数字，28 * 28辨识率的图片，输入应该是28 * 28 的矩阵，代表图片上每一个地方的像素值，因为是黑白，值为0，1


实际卷积的时候是拿一个滤波器与代表图片像素的矩阵做乘积然后相加。

![](https://github.com/sherlcok314159/ML/blob/main/Images/convalution.png)

滤波器中的值其实是learn出来的，是一个矩阵，矩阵的规模可以自己设置。

其实就是滤波器在原始矩阵上进行平移，至于一次平移多少（stride），同样是参数可以自己设置。

做好卷积之后原始如果说是28 * 28 的矩阵，经过3 * 3 的滤波器，变成 26 * 26 的矩阵了。

因为是黑白图片，所以一个矩阵就行。（channel = 1）**input_shape = (28,28,1)**，如果是彩色图片，有RGB，所以得有**三个矩阵叠在一起**形成一个**Feature Map**

![](https://github.com/sherlcok314159/ML/blob/main/Images/rgb.png)

------------------------------------------------------------------------[图片来源](https://www.youtube.com/watch?v=FrKWiRv254g&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=19)----------------------------------------------------------------------------

接下来要经过池化（max pooling），池化矩阵的规模同样是自己设置。可以试试 2 * 2。池化所做的跟滤波器一样在矩阵上平移，只不过作用是取最大。

![](https://github.com/sherlcok314159/ML/blob/main/Images/maxpooling.png)

经过池化之后，生成25 * 25的矩阵。

**卷积池化这两个过程可以不断重复多次，来减小取样（downsample）同时保留图片重要特征**

除了Maxpooling,还有其他几种，这里一并介绍

**GlobalMaxPooling**

![](https://github.com/sherlcok314159/ML/blob/main/Images/GlobalMaxPooling.png)

顾名思义，Global意味着全局，整个取一个最大值

**AveragePooling**

![](https://github.com/sherlcok314159/ML/blob/main/Images/AveragePooling.png)

跟上面MaxPooling的操作类似，唯一区别是一个求最大值，一个求平均值

**GlobalAveragePooling**

![](https://github.com/sherlcok314159/ML/blob/main/Images/GlobalAveragePooling.png)

全局求平均值

>We don't minimize total loss to find the best function.

我们采取将数据打乱并分组成一个一个的mini-batch，每个数据所含的数据个数也是可调的。关于epoch

![](https://github.com/sherlcok314159/ML/blob/main/Images/batch.png)

------------------------------------------------------------------------[图片来源](https://www.youtube.com/watch?v=FrKWiRv254g&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=19)----------------------------------------------------------------------------

将一个mini-batch中的loss全部加起来，就更新一次参数。一个epoch就等于将所有的mini-batch都遍历一遍，并且经过一个就更新一次参数。

如果epoch设为20，就将上述过程重复20遍

然后把卷积池化之后的输出全部拉成一个向量接到全连接层（fully-connected feedforward network），就是前面我们讲的[DNN](NN/dnn.md)，最后经过[softmax](../data_process/normalization.md) 输出概率

优化的时候采用的是[Adam](../optimization/GD.md)，损失函数是[交叉熵(cross-entroppy)](../loss/loss_.md)，激活函数选的为[Relu](activation.md)

****
>e.g. 实现手写数字集（Mnist）的识别

```python

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib as mpl
import sys 

# solve could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

# 不同库版本，使用此代码块查看
print(sys.version_info)
for module in mpl,np,tf,keras:
  print(module.__name__,module.__version__)

'''
sys.version_info(major=3, minor=6, micro=9, releaselevel='final', serial=0)
matplotlib 3.3.4
numpy 1.16.0
tensorflow 1.14.0
tensorflow.python.keras.api._v1.keras 2.2.4-tf
'''
# If you get numpy futurewarning,then try numpy 1.16.0

# load train and test
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# 1 Byte = 8 Bits，2^8 -1 = 255。[0,255]代表图上的像素，同时除以一个常数进行归一化。1 就代表全部涂黑。0 就代表没涂

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# CNN 的输入方式必须得带上channel，这里扩充一下维度

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# y 属于 [0,9]代表手写数字的标签，这里将它转换为0-1表示，可以类比one-hot，举个例子，如果是2

# [[0,0,1,0,0,0,0,0,0,0]……]

model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(units=10, activation="softmax"),
    ]
)

# 注意，Conv2D里面有激活函数不代表在卷积和池化的时候进行。而是在DNN里进行，最后拉直后直接接softmax就行


# kernel_size 代表滤波器的大小，pool_size 代表池化的滤波器的大小

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

history = model.fit(x_train, y_train, batch_size=128, epochs=15, validation_split=0.1) #10层交叉检验
score = model.evaluate(x_test, y_test)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Test loss: 0.03664601594209671
# Test accuracy: 0.989300012588501

# visualize accuracy and loss
def plot_(history,label):
    plt.plot(history.history[label])
    plt.plot(history.history["val_" + label])
    plt.title("model " + label)
    plt.ylabel(label)
    plt.xlabel("epoch")
    plt.legend(["train","test"],loc = "upper left")
    plt.show()

plot_(history,"acc")
plot_(history,"loss")

```
在机器学习中画精确度和loss的图很有必要，这样可以发现自己的代码中是否存在问题，并且将这个问题可视化

![](https://github.com/sherlcok314159/ML/blob/main/Images/cnn_acc.png)

![](https://github.com/sherlcok314159/ML/blob/main/Images/cnn_loss.png)


这里再细谈一下batch 和 epoch

![](https://github.com/sherlcok314159/ML/blob/main/Images/speed.png)

------------------------------------------------------------------------[图片来源](https://www.youtube.com/watch?v=FrKWiRv254g&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=19)----------------------------------------------------------------------------

由图可知，当batch数目越多，分的越开，每一个epoch的速度理所应当就会**上升**的，当batch_size = 1的时候，1 epoch 就更新参数50000次 和 batch_size = 10的时候，1 epoch就更新5000次，那么如果更新次数相等的话，batch_size = 1会花**166s**;batch_size = 10每个epoch会花**17s**，总的时间就是**17 * 10 = 170s**。其实batch_size = 1不就是[SGD](../optimization/GD.md)。随机化很不稳定，相对而言，batch_size = 10，收敛的会更稳定，时间和等于1的差不多。那么何乐而不为呢？

肯定有人要问了？随机速度快可以理解，看一眼就更新一次参数

![](https://github.com/sherlcok314159/ML/blob/main/Images/gpu.png)

为什么batch_size = 10速度和它差不多呢？按照上面来想，应该是一个mini-batch结束再来下一个，这样慢慢进行下去，其实没理由啊。

接下来以batch_size = 2来介绍一下

![](https://github.com/sherlcok314159/ML/blob/main/Images/gpu_2.png)

学过线性代数应该明白，可以将同维度的向量拼成矩阵，来进行矩阵运算，这样每一个mini-batch都在同一时间计算出来，即为平行运算

>所有平行运算GPU都能进行加速。

那么，好奇的是到底计算机看到了什么？是一个一个的数字吗？

![](https://github.com/sherlcok314159/ML/blob/main/Images/fake_digits.png)

------------------------------------------------------------------------[图片来源](https://www.youtube.com/watch?v=FrKWiRv254g&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=19)----------------------------------------------------------------------------

其实这件事情很反直觉，原以为计算机是看一张一张的图片，可是这个很难看出是单个数字而是**数字集**，那么我们试试看最大化像素


![](https://github.com/sherlcok314159/ML/blob/main/Images/cnn_learn_2.png)

------------------------------------------------------------------------[图片来源](https://www.youtube.com/watch?v=FrKWiRv254g&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=19)----------------------------------------------------------------------------

其实左下角的6其实蛮像的耶。
