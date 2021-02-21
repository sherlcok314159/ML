### Gradient Descent

**章节**
- [基础模型](#basic)
- [遇到的问题以及解决方案](#solutions)

**<div id='basic'>基础模型</div>**

一开始挑好模型集合之后，接着定义出一个损失函数，那么机器学习就变成**优化问题**，找出一组参数，使得*loss*最小。


在这个例子，我们以PM2.5值预测为例，选的是**Linear Regression**模型。


我们选的*loss function*是*RMSE(Root Mean Square Error)*

![](https://github.com/sherlcok314159/ML/blob/main/Images/root_mean_square_error.png)
              （图1）

***1.Batch-Gradient-Descent***

![](https://github.com/sherlcok314159/ML/blob/main/Images/BGD.png)

（图2 ）
BGD其实就是原始的梯度下降算法，learning rate决定了参数更新的速度。e.g.假如你在山顶上，现在你要跑到山谷处，learning rate就相对于你迈出多大步（i.e.step size）。你可以选择迭代的次数。倒数第二行伪代码的意思是你到达了一个minimum，无论往左右两边移动，都不会有太大的变化，实操时不用写上。
**<div id='solutions'>遇到的问题以及解决方案</div>**