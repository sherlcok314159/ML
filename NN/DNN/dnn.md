### DNN(Deep-Learning Neural Network)

接下来介绍比较常见的**全连接层网络（fully-connected feedfoward nerural network）**

**名词解释**

首先介绍一下神经网络的基本架构，以一个神经元为例

![](https://github.com/sherlcok314159/ML/blob/main/Images/neuron.png)

输入是一个**向量**，**权重（weights）**也是一个**矩阵**

把两个矩阵进行相乘，最后加上**偏差（bias）**，即*w1 * x1 + w2 * x2 + b*

![](https://github.com/sherlcok314159/ML/blob/main/Images/neuron_2.png)

神经元里面会有一个**激活函数（activation）**，比如[sigmoid,Relu](activation.md)等等，然后将这个结果当作未知量输入神经元所代表的函数

![](https://github.com/sherlcok314159/ML/blob/main/Images/neuron_3.png)

神经网络分为三个层次，从左到右分别为**输入层（Input Layer）**，**隐含层（Hidden Layer）**，**输出层（Output Layer）** 。当然在输出之前还得经过**softmax**，这里只是广义的网络架构。输入层和输出层没必要赘言。一个隐含层有很多神经元（neuron），神经元的数目是你可以设置的参数。

何为 **全连接（fully-connected）** 呢？

**每一个输入跟神经元必定两两相连**

![](https://github.com/sherlcok314159/ML/blob/main/Images/neuron_4.png)

------------------------------------------------------------------------[图片来源](https://bnzn2426.tistory.com/category/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%26%EB%94%A5%EB%9F%AC%EB%8B%9D%20%EA%B3%B5%EB%B6%80)----------------------------------------------------------------------------

那么，肯定有人问了

***Deep or not?***

>Universality Theorem 

>Any Deep Neuron Network can be replaced by one-hidden-layer neuron network.

于是有人说，我都可以用一层代替很多层了，那为什么还用Deep，根本没必要嘛

***Why Deep?***

确实一个很宽的隐含层可以代替很多层，但是效率必须纳入比较范围

如果浅层和深层比较，按照**控制变量法**，两个参数数量必须一样，否则结果会有偏差

举个例子，接下来我要通过图片来分辨长发男，短发男，长发女，短发女。

**浅层做法：**

![](https://github.com/sherlcok314159/ML/blob/main/Images/shallow.png)

那我们联系一下现实，现实中**长发男**不大好找诶，这就意味着这类数据量很少，train出的结果很糟。

我们都知道如果问题比较复杂的时候，可以**将问题分为几个小问题**，算法中有个思维是**分而治之（Divide and Conquer）**。这里要介绍一下**模组化（Modularization）**。

继续上面的例子，我们可以先分辨是男是女和是长发还是短发

![](https://github.com/sherlcok314159/ML/blob/main/Images/modularization.png)

------------------------------------------------------------------------[图片来源](https://www.youtube.com/watch?v=XsC9byQkUH8&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=20)----------------------------------------------------------------------------

这样的话，数据量都是够的，train出的结果不会很糟

![](https://github.com/sherlcok314159/ML/blob/main/Images/modularization_2.png)

------------------------------------------------------------------------[图片来源](https://www.youtube.com/watch?v=XsC9byQkUH8&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=20)----------------------------------------------------------------------------

通过模组化用两层神经网络可以解决数据量不足的问题，这样train出的结果比一层比起来肯定是好的多的

这也可以解释其实当我们**数据不够**的时候，用**Deep Neuron Network**其实train出的结果比其他好一点

而且神奇的是，**Modularization**在**DeepLearning**的过程中会自动从训练数据中学得，Deep的过程中会把一个复杂的问题分成若干个**Simple Function**，每一个各司其职，就像是写代码的时候会写一个函数，然后再需要用的时候，**call**一下函数名就行了

> Deep is necessary.

**通用近似原理（Universal Approximation Theorem）**

只要有一层隐含层和激活层，就能够近似拟合任意的函数，其实深度学习就是去学习去一个网络来近似出一个函数，从而最大化接近相关变量的概率分布，这也是神经网络的基础。

另外，补充一点，DNN一个好处是可以决定输出的大小（shape），只要设置隐含层的神经元即可


接下来，以手写数字识别（Mnist）为例代码实操一下

```python
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import sys
import matplotlib as mpl
import tensorflow as tf

# 不同库版本，使用本代码需要查看
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

# Load train and test
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Reshape to dense standard input
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = keras.Sequential(
    [
        keras.Input(shape=(28 * 28)),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

history = model.fit(x_train, y_train, batch_size=128, epochs=15, validation_split=0.2)
score = model.evaluate(x_test, y_test)

print("Test Loss:", score[0])
print("Test Accuracy", score[1])

# Test Loss: 0.11334700882434845
# Test Accuracy 0.9750999808311462
# 注意天下几乎没有两个相同的准确率，因为你是random出来的，在epoch之前会randomly initialize parameters

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

'''
@Tool : Tensorflow 2.x
Transform acc,val_acc above into accuracy,val_accuracy
'''
 
```
![](https://github.com/sherlcok314159/ML/blob/main/Images/dnn_accuracy_plot.png)

![](https://github.com/sherlcok314159/ML/blob/main/Images/dnn_loss_plot.png)


这里跟[CNN](cnn.md) 其实shape的处理是不一样的。在CNN中你必须再加一个维度来表示**channel**，而DNN这里是不需要的。读者也可以把print一下train_data.shape，会发现是(60000,28,28)。60000是数据的数目。输入shape应该是 28 * 28 = 784，为了与输入shape匹配，需要一开始**reshape成(60000,784)**

**注意**：千万别把 28 * 28 写出 (28,28)！否则会报错

```python
ValueError: Input 0 of layer sequential is incompatible with the layer: expected axis -1 of input shape to have value 28 but received input with shape (128, 784)
```
