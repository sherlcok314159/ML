章节
- [概述](#abstract)
- [Linear Regression](#linear)
    - [Cost function](#cost0)
    - [Gradient Descent](#gd0)
- [Logistic Regression](#logistic)
    - [Cost function](#cost1)
    - [Gradient Descent](#gd1)


**<div id='abstract'>概述</div>**


回归是机器学习中常见的任务之一，**适用于连续值的预测**，输出为一个标量，即为预测出的值

回归模型分为两类，一类是线性回归（Linear Regression），而另一类是逻辑回归(Logistic Regression)。前者是用来预测连续值，后者是用于分类问题


**<div id='linear'>Linear Regression</div>**

其实，最简单的线性模型可以是

![](https://github.com/sherlcok314159/ML/blob/main/Images/linear.png)

就是一条直线

![](https://github.com/sherlcok314159/ML/blob/main/Images/linear_model.png)

用一条直线去**拟合**整个训练集中各种**值的分布**

当然，你如果觉得太简单，可以不断**复杂**化模型，从一次调为二次，三次等等


**<div id='cost0'>Cost function</div>**

在机器学习中，我们通过定义一个模型的[损失函数](../loss/loss_.md)，通过[梯度下降](../optimization/GD.md)找到最优的参数。

![](https://github.com/sherlcok314159/ML/blob/main/Images/cost_0.png)

这里采用Least Squared Error

**<div id='gd0'>Gradient Descent</div>**

![](https://github.com/sherlcok314159/ML/blob/main/Images/gd0.png)

a是学习率，是一个需要学习的参数，具体在专门梯度下降我都写过，这里不再赘述

关于代码实操，详见[预测股票价格](stock.md)

**<div id='logistic'>Logistic Regression</div>**

linear regression 只能做连续值预测，其实从另一种角度来看，其实是可以处理分类问题的。比如按照预测出的值来判断标签，但是往往这种输出的值会有小于0和无穷大的情况，输出不稳定。但Logistic Regression是可以做分类问题的，因为它会将所有输出的值映射到[0,1]区间

![](https://github.com/sherlcok314159/ML/blob/main/Images/logistic.png)

当然，你如果熟悉[sigmoid](../NN/activation.md)，就会发现在函数上logistic regression就是把linear regression的函数放到了z上，仅此而已。关于sigmoid 相关性质，可以点击上面了解

你会发现当输入值大于0的时候，就会映射到[0.5,1]区间，同样的，当小于0的时候，会被映射到[0,0.5]，当作二元分类的时候，只有两种标签（0，1），比如你今天要做个垃圾邮件识别器，label只有两个，是 or 不是。

![](https://github.com/sherlcok314159/ML/blob/main/Images/logistic_.png)

在这里，sigmoid的输出可以作为概率


**<div id='cost1'>Cost function</div>**

![](https://github.com/sherlcok314159/ML/blob/main/Images/cost_1.png)

这里选择[Cross entropy](../loss/loss_.md) 作为loss function ，因为是二元分类问题，符合伯努力分布

可见在y = 1的时候，x 靠近 0 ，即 y = 0 loss 会越来越大。当y = 0的时候，其实图像就像是y = 1 关于（1，0）的对称。同理


**<div id='gd1'>Gradient Descent</div>**

可是，肯定有人看到这里要问了，这个Gradient Descent做不了啊，因为不能微分啊。那怎么办，把if去掉

![](https://github.com/sherlcok314159/ML/blob/main/Images/logistic_2.png)

那到底可不可行呢？

我们举个例子，当y = 1时，后面一项(1-y)就等于0，直接没了，感觉很神奇对吧

至于Gradient Descent的式子其实跟linear regression是一样的，只是注意两者模型本身不一样

