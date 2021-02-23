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

因为是黑白图片，所以一个矩阵就行。如果是彩色图片，有RGB，所以得有**三个矩阵叠在一起**形成一个**Feature Map**

![](https://github.com/sherlcok314159/ML/blob/main/Images/rgb.png)

------------------------------------------------------------------------[图片来源](https://www.youtube.com/watch?v=FrKWiRv254g&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=19)----------------------------------------------------------------------------

接下来要经过池化（max pooling），池化矩阵的规模同样是自己设置。可以试试 2 * 2。池化所做的跟滤波器一样在矩阵上平移，只不过作用是取最大。

![](https://github.com/sherlcok314159/ML/blob/main/Images/maxpooling.png)

经过池化之后，生成25 * 25的矩阵。

**卷积池化这两个过程可以不断重复多次，来减小取样同时保留图片重要特征**

然后把卷积池化之后的输出全部拉成一个向量接到全连接层（fully-connected feedforward network），就是前面我们讲的[DNN](NN/dnn.md)，最后经过[softmax](data_process/normalization.md)输出概率

**代码实操**

```python

from tensorflow