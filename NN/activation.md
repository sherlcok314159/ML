### 激活函数

**1.Sigmoid**

![](https://github.com/sherlcok314159/ML/blob/main/Images/sigmoid.png)

图像大致长这样

![](https://github.com/sherlcok314159/ML/blob/main/Images/sigmoid_2.png)

可以看出，会将所有数据映射到 **(0,1)** 区间上，若是**正数**，映射后的值大于**0.5**；若为**负数**，映射后的值小于**0.5**；若为**0**，则为**0.5**

可是，**Sigmoid**在**DeepLearning**的时候会随着层数的增加**loss**反而飙升

![](https://github.com/sherlcok314159/ML/blob/main/Images/vanish.png)

------------------------------------------------------------------------[图片来源](https://www.youtube.com/watch?v=xki61j7z-30&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=16)----------------------------------------------------------------------------

![](https://github.com/sherlcok314159/ML/blob/main/Images/vanish_2.png)

------------------------------------------------------------------------[图片来源](https://www.youtube.com/watch?v=xki61j7z-30&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=16)----------------------------------------------------------------------------

其实在**input**这边，gradient很小，参数几乎还是随机更新；而**output**这边，gradient很大，参数更新很快，几乎就收敛了（converge）。导致**input**这边参数还在随机更新的时候，**output**这边就已经根据**random**出来的参数找到了**local minimum**，然后你就会观察说糟糕了，**loss**下降的速度很慢

![](https://github.com/sherlcok314159/ML/blob/main/Images/vanish_3.png)

------------------------------------------------------------------------[图片来源](https://www.youtube.com/watch?v=xki61j7z-30&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=16)----------------------------------------------------------------------------

因为**Sigmoid**将负无穷到正无穷的数据**硬压**到（0，1），当**input**很大的时候，其实**output**的结果**变化很小**。本来影响就很小，你又叠了很多层，影响被迫**衰减**，就造成了**input**几乎对loss产生不了什么影响。

**2.ReLu**

为了解决上面的问题，应该将激活函数换成**ReLu**


![](https://github.com/sherlcok314159/ML/blob/main/Images/relu.png)

图像大致长这样

![](https://github.com/sherlcok314159/ML/blob/main/Images/relu_2.png)

当**input**小于等于0的时候，直接变成0。这就意味着有些神经元（neuron）会对整个神经网络没有任何的影响，所以这些神经元是可以被拿掉的

![](https://github.com/sherlcok314159/ML/blob/main/Images/relu_3.png)

------------------------------------------------------------------------[图片来源](https://www.youtube.com/watch?v=xki61j7z-30&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=16)----------------------------------------------------------------------------

那些没用的神经元直接去掉

![](https://github.com/sherlcok314159/ML/blob/main/Images/relu_4.png)


------------------------------------------------------------------------[图片来源](https://www.youtube.com/watch?v=xki61j7z-30&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=16)----------------------------------------------------------------------------

这些就**不会存在很小**的gradient，就有效地减免一开始**Sigmoid**中产生的问题

其实ReLu还有其他的形式

![](https://github.com/sherlcok314159/ML/blob/main/Images/relu_others.png)


------------------------------------------------------------------------[图片来源](https://www.youtube.com/watch?v=xki61j7z-30&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=16)----------------------------------------------------------------------------

**3.Maxout**

其实**ReLu**是**Maxout**中的一种。

拿下图举个例子

![](https://github.com/sherlcok314159/ML/blob/main/Images/maxout.png)

你想如果在ReLu中，负的没有，输入是不是跟这个一样呢？

那我们再从理论解释一下这个，如果将输送给**w2**的权重和偏差全部设为0，得出的图像是不是就是x轴，将**w1**的权重设为1，偏差设为0，是不是就是ReLu了呢？

![](https://github.com/sherlcok314159/ML/blob/main/Images/maxout_2.png)

------------------------------------------------------------------------[图片来源](https://www.youtube.com/watch?v=xki61j7z-30&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=16)----------------------------------------------------------------------------

但是**Maxout**的权重和偏差不是固定的，它是可以**通过不同的数据学出不同的曲线**

综上所述，**ReLu** 和 **Maxout** 都是根据不同的自身的判别条件而自动**剔除**一些数据，从而产生瘦的神经网络，**不同的数据会产生不同的神经网络架构**

**4.Tanh**

![](https://github.com/sherlcok314159/ML/blob/main/Images/tanh.png)

图像大致长这样

![](https://github.com/sherlcok314159/ML/blob/main/Images/tanh_2.png)

tanh的特点：

**1.梯度更新更快** 由上图可以发现，当输入变化很大，原本的sigmoid输出变化很少，这不利于梯度更新。而tanh可以有效的解决这个问题

**2.不变号** 正就被映射成正的，负的被映射成负的，零就被映射成零。另外，它过原点。

**5.Softplus**

![](https://github.com/sherlcok314159/ML/blob/main/Images/softplus_2.png)

![](https://github.com/sherlcok314159/ML/blob/main/Images/softplus.png)

它相对于ReLu相对来说更加平滑，不过同样是单侧抑制


**6.Swish**

![](https://github.com/sherlcok314159/ML/blob/main/Images/swish_2.png)

![](https://github.com/sherlcok314159/ML/blob/main/Images/swish.png)

主要优点

**1.** 无界性有助于慢速训练期间，梯度逐渐接近0并导致饱和；

**2.** 导数恒大于0

**3.** 平滑度在优化和泛化有很大作用

