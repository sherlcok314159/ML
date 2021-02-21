### Gradient Descent

**章节**
- [BGD](#bgd)
- [遇到的问题以及解决方案](#solutions)


一开始挑好模型集合之后，接着定义出一个损失函数，那么机器学习就变成**优化问题**，找出一组参数，使得**Loss**最小。


在这个例子，我们以PM2.5值预测为例，选的是**Linear Regression**模型。


我们选的**Loss Function**是**RMSE(Root Mean Square Error)**

![](https://github.com/sherlcok314159/ML/blob/main/Images/root_mean_square_error.png)

**<div id='bgd'>Batch-Gradient-Descent</div>**


```python
def BGD(T,lr,w_init):
    #initialize 
    w = w_init
    for i in range(T):
        w = w - lr * gradient(w)
    return w
```
**参数解释**
<!-- **** -->
**BGD**其实就是原始的梯度下降算法，**Learning Rate(lr)**决定了参数更新的速度。e.g.假如你在山顶上，现在你要跑到山谷处，**Learning Rate**就相对于你迈出多大步（i.e.Step Size）。

关于梯度，继续上面的那个例子，其实就是下山的时候你往哪个方向走。梯度是一个向量

![](https://github.com/sherlcok314159/ML/blob/main/Images/gradient.png)

![](https://github.com/sherlcok314159/ML/blob/main/Images/gd_0.png)
>[图片来源](https://rasbt.github.io/mlxtend/user_guide/general_concepts/gradient-optimization/)
微分从图形上来讲就是某点处的斜率，其实方向就是沿斜率方向向下

迭代次数的设定其实没什么好说的。我们来讨论一下**Learning Rate**。
****

**问题探讨**

从上图中不难发现我们卡在**Local Minimum**了，我们想要到达**Global Minimum**，有人可能会想，卡住了没关系










**<div id='solutions'>遇到的问题以及解决方案</div>**