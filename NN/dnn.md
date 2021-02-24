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

何为**全连接（fully-connected）**呢？

**每一个输入跟神经元必定两两相连**

![](https://github.com/sherlcok314159/ML/blob/main/Images/neuron_4.png)

那么，肯定有人问了

***Deep or not?***

>Universality Theorem 

>Any Deep Neuron Network can be represented by one-hidden-layer neuron network.

于是有人说，我都可以用一层代替很多层了，那为什么还用Deep，根本没必要嘛

***Why Deep?***

确实一个很宽的隐含层可以代替很多层，但是效率必须纳入比较范围

如果浅层和深层比较，按照**控制变量法**，两个参数数量必须一样，否则结果会有偏差

举个例子，接下来我要通过图片来分辨长发男，短发男，长发女，短发女。

**浅层做法：**

![](https://github.com/sherlcok314159/ML/blob/main/Images/shallow.png)

那我们联系一下现实，现实中**长发男**不大好找诶，这就意味着这类数据量很少，train出的结果很糟
