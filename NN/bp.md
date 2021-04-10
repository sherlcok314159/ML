### 后向传播算法（BackPropagation）

首先介绍一下链式法则

![](https://github.com/sherlcok314159/ML/blob/main/Images/functions.png)

假如我们要求z对x1的偏导数，那么势必得先求z对t1的偏导数，这就是链式法则，一环扣一环

![](https://github.com/sherlcok314159/ML/blob/main/Images/chain_rule.png)

BackPropagation（BP）正是基于链式法则，通常我们在深度学习会通过梯度下降寻找最优解，那么梯度如何算呢？在普通的结构中很容易，可是神经网络叠了很多层之后又该如何算呢？我们可以类比上面的式子，把hidden layer里面的结果都当做是中间变量，地位与t1相当。那么，我们要求y对某个x的偏导数，先求出y对离它最近的结果的偏导数，然后不断往后，最终达到想要找的x。而BP会将每个节点算出的偏导数保留，等到想要的时候直接拿出来用，省去多余的无效运算。

接着我们用PyTorch来实操一下后向传播算法，PyTorch可以实现自动微分，requires_grad 的意思是最终对这个tensor所进行的操作，这样我们进行BP的时候，直接用就好。不过默认情况下是False，需要我们手动设置。

```python
import torch

x = torch.ones(3,3,requires_grad = True)
t = x * x + 2 
z = 2 * t + 1
y = z.mean()
```

接下来我们想求y对x的微分，需要注意这种用法只能适用于y是标量而非向量

```python
y.backward()
print(x.grad)
```

![](https://github.com/sherlcok314159/ML/blob/main/Images/jacobi.png)

所以当y是向量形式的时候我们可以自己来算，如

```python
x = torch.ones(3,3,requires_grad = True)
t = x * x + 2
y = t - 9
```

如果计算y.backward()会报错，因为是向量，所以我们需要手动算v，这里就是y对t嘛，全为1，注意v的维度就行。

```python
v = torch.ones(3,3)
y.backward(v)
print(x.grad)
```