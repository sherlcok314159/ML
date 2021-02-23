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

**卷积池化这两个过程可以不断重复多次，来减小取样同时保留图片重要特征**

>We don't minimize total loss to find the best function.

我们采取将数据打乱并分组成一个一个的mini-batch，每个数据所含的数据个数也是可调的。关于epoch

![](https://github.com/sherlcok314159/ML/blob/main/Images/batch.png)

------------------------------------------------------------------------[图片来源](https://www.youtube.com/watch?v=FrKWiRv254g&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=19)----------------------------------------------------------------------------

将一个mini-batch中的loss全部加起来，就更新一次参数。一个epoch就等于将所有的mini-batch都遍历一遍，并且经过一个就更新一次参数。

如果epoch设为20，就将上述过程重复20遍

然后把卷积池化之后的输出全部拉成一个向量接到全连接层（fully-connected feedforward network），就是前面我们讲的[DNN](NN/dnn.md)，最后经过[softmax](data_process/normalization.md)输出概率

优化的时候采用的是[Adam](optimization/GD.md)，损失函数是[交叉熵(cross-entroppy)](loss/loss_.md)，激活函数选的为[Relu](NN/activation.md)

****
**代码实操**

```python
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np

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
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(units=10, activation="softmax"),
    ]
)

# 注意，Conv2D里面有激活函数不代表在卷积和池化的时候进行。而是在DNN里进行，最后拉直后直接接softmax就行


# kernel_size 代表滤波器的大小，pool_size 代表池化的滤波器的大小

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=128, epochs=15, validation_split=0.1) #10层交叉检验
score = model.evaluate(x_test, y_test)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Test loss: 0.03664601594209671
# Test accuracy: 0.989300012588501
```
