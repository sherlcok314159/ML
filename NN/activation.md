### 激活函数

**1.Sigmoid**

![](https://github.com/sherlcok314159/ML/blob/main/Images/sigmoid.png)

图像大致长这样

![](https://github.com/sherlcok314159/ML/blob/main/Images/sigmoid_2.png)

可以看出，会将所有数据映射到**(0,1)** 区间上，若是**正数**，映射后的值大于**0.5**；若为**负数**，映射后的值小于**0.5**；若为**0**，则为**0.5**

可是，**Sigmoid**在**DeepLearning**的时候会随着层数的增加**loss**反而飙升

![](https://github.com/sherlcok314159/ML/blob/main/Images/vanish.png)

------------------------------------------------------------------------[图片来源](https://www.youtube.com/watch?v=xki61j7z-30&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=16)----------------------------------------------------------------------------

![](https://github.com/sherlcok314159/ML/blob/main/Images/vanish_2.png)

------------------------------------------------------------------------[图片来源](https://www.youtube.com/watch?v=xki61j7z-30&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=16)----------------------------------------------------------------------------

其实在**input**这边，gradient很小，参数几乎还是随机更新；而**output**这边，gradient很大，参数更新很快，几乎就收敛了（converge）。导致**input**这边参数还在随机更新的时候，**output**这边就已经根据**random**出来的参数找到了**local minimum**，然后你就会观察说糟糕了，**loss**下降的速度很慢

![](https://github.com/sherlcok314159/ML/blob/main/Images/vanish_3.png)

------------------------------------------------------------------------[图片来源](https://www.youtube.com/watch?v=xki61j7z-30&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=16)----------------------------------------------------------------------------

因为**Sigmoid**将负无穷到正无穷的数据**硬压**到（0，1），当**input**很大的时候，其实**output**的结果**变化很小**。本来影响就很小，你又叠了很多层，影响被迫**衰减**，就造成了**input**几乎对loss产生不了什么影响。

**2.Relu**

为了解决上面的问题，应该换一个激活函数


![](https://github.com/sherlcok314159/ML/blob/main/Images/relu.png)

图像大致长这样

![](https://github.com/sherlcok314159/ML/blob/main/Images/relu_2.png)

