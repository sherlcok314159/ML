### Gradient Descent

**章节**
- [BGD](#bgd)
- [SGD](#sgd)
- [Better](#better)
    - [Momentum](#mom)
    - [RMSProp](#rms)
    - [Adam](#adam)
    - [Adagrad](#adag)

一开始挑好模型集合之后，接着定义出一个损失函数，那么机器学习就变成**优化问题**，找出一组参数，使得**Loss**最小。


在这个例子，我们以PM2.5值预测为例，选的是**Linear Regression**模型。


我们选的**Loss Function**是**RMSE(Root Mean Square Error)**

![](https://github.com/sherlcok314159/ML/blob/main/Images/root_mean_square_error.png)

**<div id='bgd'>Batch-Gradient-Descent</div>**


```python
def BGD(T,lr):
    #initialize 
    w = 0
    for i in range(T):
        w = w - lr * gradient(w)
    return w
```
**参数解释**
<!-- **** -->
**BGD**其实就是原始的梯度下降算法，**Learning Rate(lr)** 决定了参数更新的速度。e.g.假如你在山顶上，现在你要跑到山谷处，**Learning Rate** 就相对于你迈出多大步（i.e.Step Size）。

梯度是所有的**w**微分之后所组成的一个向量

![](https://github.com/sherlcok314159/ML/blob/main/Images/gradient.png)

![](https://github.com/sherlcok314159/ML/blob/main/Images/gd_0.png)


有人会问为什么是**w**减，而不是**w**加呢？

斜率若为正，按照直觉知道左边函数值小，那应该往左走，就是**w**减去一个正数。

斜率若为负，函数值是左大右小，那么应该往右走，就是**w**加上一个正数。


迭代次数的设定其实没什么好说的。我们来讨论一下**Learning Rate**。

**A.** 若**lr**稍微大点会如何呢？

![](https://github.com/sherlcok314159/ML/blob/main/Images/gd_1.png)

我们会发现上面四步我们两步就到了，是不是很激动？那我们更大一点会如何呢？

![](https://github.com/sherlcok314159/ML/blob/main/Images/gd_2.png)

直接烂掉，**如果太大，会发生大幅度的偏离** ，如果是实操，输出会是**nan**

**B.** 若**lr**稍微小一点呢？

这里就不画图了。不过是走的时间多一点，更新参数次数多一点。只要不是小碎步，你最终还是可以走到全局最优的。


**注意**

**它走的方向只是在图像上往左还是往右**，本质上是横坐标的**加减变化**，而不是按照斜率去走的。梯度可以想象成你走之前往旁边看一眼，看看是不是更低一点。然后你再决定往左还是往右走。 

**w**没有必要设成零向量，还得看看维度，直接设成0，第一次迭代结束就是一个向量了

不同人的**Loss Function**的设定可能不一样，因为可以是预测值减去真实值，也可以是真实值减去预测值，因为是均方偏差。你看别人的代码需要注意这一点，**因为微分后不同的写法决定梯度是否差一个负号**

所优化的函数一定要**可微！可微！可微！**，那遇到不可微的怎么办呢？

比如说 [ReLu](../NN/activation.md)，既然在原点处不可微，那就直接不看好了，因为输入很小机率是0

接下来讨论一下跟它很像的SGD

**<div id='sgd'>Stochastic-Gradient-Descent</div>**

![](https://github.com/sherlcok314159/ML/blob/main/Images/sgd.png)

------------------------------------------------------------------------[图片来源](https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week4/gradient_descent/?activate_block_id=block-v1%3AMITx%2B6.036%2B1T2019%2Btype%40sequential%2Bblock%40gradient_descent)----------------------------------------------------------------------------

**注意**，虽然BGD和SGD看起来很像，但其实BGD是将整个数据看过一遍；而SGD只是从Dataset中Sample出某一个点。所以两者的梯度都不一样，前者梯度会带求和符号，而SGD就单个点的梯度。

![](https://github.com/sherlcok314159/ML/blob/main/Images/sgd_2.png)

可以看出，SGD收敛的十分**剧烈**

***BGD VS SGD？***

SGD is better in 3 reasons:

**1.Fast Computation**。虽然BGD很稳定，但是它必须看过整个数据才能进行更新参数，而数据量太大，会花很长时间。

**2.Beat Local Minimum** BGD有时候会被卡在局部最优，而SGD可能有望跳出来

**3.Avoid Overfitting** 有的时候我们选择SGD是不想要过拟合，BGD的拟合效果是好于SGD的

****

细心的读者会发现图中有一个 **Convex** ，意思是凸函数。类如**f(x) = (x - 2) ^ 2**

理想总是很丰满，那是因为你没有经历过社会（数据）的毒打，如果不是 **Convex** ，会是如何呢？

****

**<div id='better'>问题探讨</div>**

![](https://github.com/sherlcok314159/ML/blob/main/Images/gd.png)

从上图中不难发现我们卡在**Local Minimum**了，我们想要到达**Global Minimum**，有人可能会想，卡住了没关系，我迭代次数多一点，让他慢慢走出来不就行了？细心观察就会明白，小步伐的时候，严格按照**Gradient** 指引的方向走，左右两边是相反方向。i.e. 你会一直往返直到结束

那么，如何解决这个问题呢？

**1.依靠惯性**

假如他卡在那个**Local**的地方了，他如果具有惯性，能够自己滚出来，会不会有机会到达**Global**？

**<div id='mom'>Momentum</div>**

```python
def momentum(T,lr,k):
    w = 0
    v = 0
    for i in range(T):
        v = k * v - lr * gradient(w)
        w = w + v
    return w 
```



![](https://github.com/sherlcok314159/ML/blob/main/Images/momentum.png)

当然，这个方法还是比较看人品的，参数**k**是需要你自己调的。是时候表演真正的技术（人品）了，一般在 **[0.5,0.9]** 的范围调，来决定惯性对整个式子的影响程度。

像我一直脸黑，当然不能靠这种。

**<div id='rms'>2.RMSProp</div>**

![](https://github.com/sherlcok314159/ML/blob/main/Images/rmsprop.png)

其他跟**BGD**一样

**<div id='adam'>3.Adam</div>**


**Adam = RMSProp + Momentum**

![](https://github.com/sherlcok314159/ML/blob/main/Images/adam.png)


**4.Adagrad**

可以将缩写理解为**Adaptive Gradient Descent**，意思就是**lr**会随着参数的更新一起对应更新

与**BGD**相比，这个不一样罢了

![](https://github.com/sherlcok314159/ML/blob/main/Images/adagrad.png)

里面还加了一个极小的数，防止分母为0，一般取1.0e-7




