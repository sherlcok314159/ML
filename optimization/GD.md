### Gradient Descent

**章节**
- [基础模型](#basic)
- [遇到的问题以及解决方案](#solutions)

**<div id='basic'>基础模型</div>**

一开始挑好模型集合之后，接着定义出一个损失函数，那么机器学习就变成**优化问题**，找出一组参数，使得*loss*最小。


在这个例子，我们以PM2.5值预测为例，选的是**Linear Regression**模型。


我们选的*loss function*是*RMSE(Root Mean Square Error)*

![](https://github.com/sherlcok314159/ML/blob/main/Images/root_mean_square_error.png)

***1.Batch-Gradient-Descent***

```python
def BGD(T,lr,w_init):
    #initialize 
    w = w_init
    for i in range(T):
        w = w - lr * gradient(w)
    return w
```
BGD其实就是原始的梯度下降算法，learning rate(lr)决定了参数更新的速度。e.g.假如你在山顶上，现在你要跑到山谷处，learning rate就相对于你迈出多大步（i.e.step size）。你可以选择迭代的次数。

关于梯度，继续上面的那个例子，其实就是下山的时候你往那个方向走。梯度是一个向量


![](https://github.com/sherlcok314159/ML/blob/main/Images/gradient.png)


**<div id='solutions'>遇到的问题以及解决方案</div>**