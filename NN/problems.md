![](https://github.com/sherlcok314159/ML/blob/main/Images/Recipe_for_DNN.jpg)


章节
- [Train Problems](#train)
    - [Change Activation](#activation)
    - [Update Learning Rate](#lr)
- [Test Problems](#test)
    - [Dropout](#drop)
    - [EarlyStopping](#stop)
    - [Regularization](#regular)
- [参考文献](#references)

**<div id='train'>训练出问题</div>**

首先从**train**开始，有些时候train的时候结果就不怎么样，然后没看就**直接test**，然后结果烂掉了。就说是**过拟合（overfitting）**。这个说法显然是错误的，因为首先你必须在训练集上performances很好，test烂掉才能说是过拟合。train的时候就失败了，只能说是**欠拟合（underfitting）**

欠拟合的原因最基本的是**模型过于简单（参数少，偏差大）**，这种情况可以**适当复杂化**，比如说回归问题，一开始函数是一次函数，可以往三次，四次上调。

可以比喻成打靶，**bias很大**意思就是根本**没有瞄准靶打**，但是Varriance小，受到外界影响小。

**<div id='activation'>Change Activation</div>**
有些时候不同情形需要用不同的激活函数，另外，有些函数存在本身超越不了的问题，比如sigmoid(Vanish Gradient Descent)，在[Activation](../NN/activation.md)。所以需要根据具体情况来**调整激活函数**


**<div id='lr'>Update Learning Rate</div>**

在[Gradient Descent](../optimization/GD.md)中已经具体探讨过了，BGD大概率会卡在**Local minimum**的地方，这个时候可以选择**adagrad 或者 adam** 不断更新learning rate

**<div id='test'>测试出问题</div>**

train的结果好，测试的结果烂掉了，那就是过拟合了。同样取打靶，复杂的模型bias是小的，也就是**瞄准靶心打的**，但是模型过于复杂，可以想成受到各个方面的影响就越大，如风速，最后还是打不中，这种情况是varriance（方差大）

**<div id='drop'>Dropout</div>**

![](https://github.com/sherlcok314159/ML/blob/main/Images/dropout.png)


由图可知，**不一定所有神经元**都会被训练到，每一个神经元都有p%被去掉，所以训练的时候很有可能就**不是同一个**模型，这或许可以从直觉上解释为什么Dropout**对train反而不友好**，而通过这种去除的方式可以一定程度**防止过拟合**，会让**Test结果好看**一点

需要注意的是一般使用Dropout的时候**batch_size = 1**，同时有些**参数**在神经网络中是**共用**的

另外，如果不加Dropout的要想实现Dropout的结果，必须把**权重（weights）都乘上（1-p%）**。其实这里有点奇怪，那我这里举两个weights为例

![](https://github.com/sherlcok314159/ML/blob/main/Images/Dropout.jpg)


会发现结果竟然惊人的相同

在train的时候可以选择去掉，而在**test**的时候是**没有Dropout**的，这个时候进行如上操作

If a weight = 1 by training,set w = 0.5 for testing.(Dropout rate is 50%.)

其实，Dropout是一种**Ensemble**


![](https://github.com/sherlcok314159/ML/blob/main/Images/dropout_ensemble.png)


在模型比较复杂的时候，我们常常会从数据集中**Sample**出小部分的data来训练，每次sample出的data训练的结果都不一样，也许单个varriance都比较大，这个时候我们**取个平均**，结果就会好很多，那么，为什么说Dropout与Ensemble有关呢？

![](https://github.com/sherlcok314159/ML/blob/main/Images/dropout_2.jpg)


虽然batch_size 都为1，但是，每次被训练的神经网络是不一样的，这可以类似地认为从一个大模型里面sample出一个小模型，这里不要与randomly initialize parameters搞混，记得我们在[CNN](CNN/cnn.md)中说过epoch和batch_size。那里只是随机初始化参数。

其实很多时候一个batch让人比较不安，**data会不会太少**，其实不用担心的，因为参数大部分是共用的，如果有一个weight在几个模型里都没有dropout，那么，**它会被几个模型train**。


**<div id='stop'>EarlyStopping</div>**

其实很多时候看图发现当train_accuracy越来越大的时候，validation_loss其实已经**不太会往下降**了，这个时候可以**提前停下来**

示例如下

```python
from tensorflow.keras.callbacks import EarlyStopping

# patience越大对不下降越不敏感，就越有耐心
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

model.fit(x, y, validation_split=0.2, callbacks=[early_stopping])
```
>**Patience**: **number** of **epochs** with **no improvement** after which training will be stopped.

简单来说就是没有进步的epoch个数

**<div id='regular'>Regularization</div>**

其实关于regularization，惩罚模型过拟合，**在模型的简单程度和逼近数据的程度之间做权衡（trade-off）**

**L1 regularization**

![](https://github.com/sherlcok314159/ML/blob/main/Images/l1.jpg)


![](https://github.com/sherlcok314159/ML/blob/main/Images/l2.png)

可以看到regularization只跟weights有关，因为**bias**只会涉及到图像**上下移**，不用考虑

最终找到不仅使loss function 变小，而且还要靠近0

![](https://github.com/sherlcok314159/ML/blob/main/Images/trade_off.jpg)

通过regularization来trade-off掉原来参数的影响，如果很受原来参数影响，模型又过于复杂，很容易受到其他因素影响，那就很容易过拟合了。lambda越大，右边越重，参数的影响就越小，而当lambda太大时，类比f(x) = c，无论你feature怎么变，我都不会care，导致欠拟合。使用regularization使得函数变得平滑，不容易过拟合

通常来讲，**L1**无论如何都会减去一个**固定**的值，而**L2**会根据不同情况来减小。所以当**W很大**的时候，**L1**减去的值还是固定的，会**下降的很慢**，而**L2**发现weight很大，也会**下降的比较快**，这个时候选择**L2**。而当**W很小**的时候，**L1依旧减去固定值**，**L2**发现很小，就**几乎卡住**了，这个时候选择L1

其实在DeepLearning里面，regularization跟EarlyStopping其实功能差不多，没有在SVM中来的那么重要

那么，如何用代码实操呢？

```python
from tensorflow.keras import layers
from tensorflow.keras import regularizers

layer = layers.Dense(
    units=64,
    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5)
)
```

或者

```python
layer = tf.keras.layers.Dense(5, kernel_initializer='ones', # 这只是示例
                              kernel_regularizer=tf.keras.regularizers.l1(0.01),
                              activity_regularizer=tf.keras.regularizers.l2(0.01))
tensor = tf.ones(shape=(5, 5)) * 2.0
out = layer(tensor)
# The kernel regularization term is 0.25
# The activity regularization term (after dividing by the batch size) is 5
print(tf.math.reduce_sum(layer.losses))  # 5.25 (= 5 + 0.25)
```

或者自定义一个
```python
def my_regularizer(x):
    return 1e-3 * tf.reduce_sum(tf.square(x))

```

若要了解更多，可以去keras官网看一下[具体接口](https://keras.io/api/layers/regularizers/)

***
**<div id='references'>参考文献</div>**

https://www.youtube.com/watch?v=xki61j7z-30&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=17&ab_channel=Hung-yiLee