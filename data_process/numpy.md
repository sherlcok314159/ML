### Introduction to numpy 
```python
import numpy as np
```
***
*1.创建数组*
```python
#维数与中括号对数有关
print(np.array([1,2,3]))
#1 dimension [1,2,3]
print(np.array([[1,2,3]]))
#2 dimensions [[1,2,3]]
```


*2.转置矩阵*
```python
#不改变维度，一维不变
a = np.array([1,2,3])
print(np.transpose(a))
#[1,2,3]
b = np.array([[1,2,3]])
print(b.T)
#[[1]
# [2]
# [3]]
```

*3.矩阵属性*
```python
#1 dimension
a = np.array([1,2,3])
print(np.shape(a))
print(a.shape)
#需要注意有两种方式表达
#(3,) 
#(3,) 
print(a.shape[0])
#3
print(type(a.shape))
#<class 'tuple'>,元组是支持索引的

#2 dimensions
b = np.array([[1,2,3],[4,5,6]])
print(np.shape(b))
#(2,3)
#改变属性
c = b.reshape(3,2)
print(c)
#[[1 2]
# [3 4]
# [5 6]]
#需要注意的是，原先的矩阵不会发生改变，需要重新赋一个
```

*4.矩阵相乘*
```python
a = np.array([[1,2,3]])
b = np.array([[4,5,6]])
print(np.dot(a.T,b))
print(a.T @ b)
#两者结果一样
#[[ 4  5  6]
# [ 8 10 12]
# [12 15 18]]
#注意满足列数等于行数
data = np.transpose(np.array([[1, 2], [1, 3], [2, 1], [1, -1], [2, -1]]))
th = np.array([[1], [1]])
th0 = -2
def signed_dist(x, th, th0):
    return (np.dot(th.T, x) + th0) / length(th)
#注意，这里是矩阵整体运算，而不是单指某个元素
```

*5.返回判断正负性*
```python
#正数返回1，零返回0，负数返回-1
print(np.sign(1))
# 1
print(np.sign(0))
# 0
print(np.sign(-1))
# -1
```

*6.求和*
```python
a = np.array([[1,2,3],[4,5,6]])
print(np.sum(a))
#21 不加参数为全部求和
print(np.sum(a,axis = 0))
#[5,7,9]
#看最外面的中括号，里面的元素为[1,2,3],[4,5,6],两个列表之间进行操作
print(np.sum(a,axis = 1))
#[6,15] 
#看最里面的中括号，里面的元素是不是一个为1，2，3.另一个是4，5，6，那元素之间进行求和，两者互不干扰
#同样的axis原则适用于np其他函数
test = [[ True  True False False  True]]
print(np.sum(test,axis = 1))
#[3]
print(np.sum(test,axis = 1,keepdims = True))
#[[3]]
#记住sum之后会清除axis确定的维度，如果需要保留，需要将keepdims = True
```

*7.索引与切片*
```python
a = np.array([[1,2,3],[4,5,6]])
print(a[0])
#[1,2,3] 
b = np.array([[1,2,3,1,4]])
print(b[2])
#IndexError: index 2 is out of bounds for axis 0 with size 1

#整数索引时以里面的小列表为元素，另外注意行向量于列向量的区别
print(a[0:1])
print(a[0:1,])
#[[1,2,3]]
#[[1,2,3]]
print(a[:,2:])
#[[3]
# [6]]
#注意，如果中间加逗号，则意味着前面对行进行操作，后面对列进行操作
print(a[:,0])
#[1,4]
#注意，如果单独用整数索引或者切片，数组会保持原来的维度，但如果两者混用，则会导致降维

data = np.array([[1,2,3],[4,5,6]])
print(data[:,[0,2,1]])
#[[1 3 2]
# [4 6 5]]
#需要注意索引可以是列表，这样就等同于更改原数据进行一个洗牌，评估算法的时候更精确，同样是保持原型的
```

*8.行向量与列向量*

```python
#row vector
a = np.array([[1,2,3,4]])
#column vector
b = np.array([[1],[2],[3]])
#接收一个列表变为行向量
t = np.array([value_list])
#接收一个列表变为列向量
q = t.T or np.transpose(t)
```

*9.数组的长度*
```python
a = np.array([[1,2,3]])
print(np.sum(a * a)**0.5)
#3.7416573867739413 
#与求向量的模类似
```

*10.布尔值判断*
```python
g = np.array([[1.,-1.,1.,1.,-1.]])
labels = np.array([[1.,-1.,-1.,-1.,-1.]])
print(g == labels)
#[[ True  True False False  True]]
#需要注意的是，numpy对布尔值也可以求和，
print(np.sum(g == labels))
#3
True = 1,False = 0
#还可以一对多布尔值判断
print(np.array([[1,2,3,4]]) == 1)
#[[ True False False False]]

if np.array([[1]]) == 1:
    print("yep!")
#yep!
#需要注意的是虽然是一个列表，在条件判断上可以等价为单独的True or False    

#如果判断的两个列表长度不一样，会单独False，而且会警告
t1 = np.array([[1, 2, 3]])
t2 = np.array([[1, 2, 3, 4]])
print(t1 == t2)
# DeprecationWarning: elementwise comparison failed; this will raise an error in the future.
#  print(t1 == t2)
#  False

t3 = np.array([[1, 2, 3, 4]])
print(np.equal(t2,t3))
#[[ True  True  True  True]]
#同型矩阵用equal

#注意如果是矩阵与单元素判断，矩阵元素个数必须等于1

if np.array([[1]]) > 0:
   print(1)
#1
if np.array([[1,2]]) > 0:
   print(2)
#ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()  
```

*11.返回最大值索引*
```python
a = np.argmax(np.array([[1,2,3,4,5,6]]))
print(a)
#5
```

*12.分割处理*
```python
a = np.array([[1,2,3,4,5],[6,7,8,9,0]])
print(np.array_split(a,5,axis = 1))

#[array([[1],
#        [6]]), array([[2],
#        [7]]), array([[3],
#        [8]]), array([[4],
#        [9]]), array([[5],
#        [0]])]
#axis不再赘言，split之后会产生列表，跟Python语法一样的
```

*13.合成*

```python
a = np.array([[1]])
b = np.array([[2]])
print(np.concatenate((a,b),axis = 1))
#[[1 2]]
b = np.array([[1,2,3,4,5],[6,7,8,9,0]])
b_div = np.array_split(b,5,axis = 1)

print(np.concatenate(b_div[0:3]+b_div[4:],axis = 1))
#[[1 2 3 5]
# [6 7 8 0]]
```

*14.生成特定向量*
```python
# 没有加维度，默认一维
print(np.zeros(5))
# [0. 0. 0. 0. 0.]
# 可以设置维度
print(np.zeros((2, 1)))
#[[0.]
# [0.]]
# 改变元素类型
print(np.zeros((2, 1), dtype=int))
#[[0]
# [0]]
#需要十分小心和注意的是，如果要几X几的矩阵，必须套一个括号！！！(2,1)
```

*15.一对多属性*
```python
#矩阵所有元素相加一个数
test = np.array([[1,2,3]])
print(test + 1)
#[[2 3 4]]

a = np.array([[1]])
b = np.array([[1, 2, 3]])
print(a + b)
#[[2 3 4]]

#矩阵所有元素与一个值做判断
print(test == 2)
#[[False True False]]
```

*16.注意点*
```python
data = [[1,2,3],[4,5,6]]
print(data[:,0:1])
#TypeError: list indices must be integers or slices, not tuple

#因为你没有用np.array
data = np.array([[1,2,3],[4,5,6]])
```

```python
#注意计算机中的运算逻辑，如果拿不准，一定要套括号   
def sd(th, th0, x):
    return np.dot(th, x) + th0 / (np.sum(th * th) ** 0.5)
#[[2.17157288]]
#[[-0.82842712]]
#[[3.17157288]]

def sd(th, th0, x):
    return (np.dot(th, x) + th0) / (np.sum(th * th) ** 0.5)
#[[0.70710678]]
#[[-1.41421356]]
#[[1.41421356]]
```
```python
#需要注意不同列表推导式生成的是列表不错，但两者具有不同的维度
x = [range(100)]
y = [i for i in range(100)]
print(type(x))
print(type(y))
print(np.shape(x))
print(np.shape(y))
#<class 'list'>
#<class 'list'>
#(1, 100)
#(100,)
```
*17.随机浮点数生成*
```python
print(np.random.rand(3,2))
#[[0.5488135  0.71518937]
# [0.60276338 0.54488318]
# [0.4236548  0.64589411]]

#rand的用法与zeros是一样的，生成几叉几的，但不同的是zeros需要套括号，而rand不用

print(np.random.rand(4, 1))
print(np.zeros((4, 1)))

#如果一个数，就是一维的列表
print(np.random.rand(3))
#[0.12277558 0.57764143 0.59843161]

#如果要求生成的随机数可以被预测，可以用np.random.seed(k)当然k可以任取
np.random.seed(0)
print(np.random.rand(3,1))
#[[0.52372051]
# [0.6603495 ]
# [0.64741481]]

np.random.seed(0)
print(np.random.rand(3,1))
#[[0.52372051]
# [0.6603495 ]
# [0.64741481]]

#你会发现生成的随机数是一样的
```

*18.洗牌*
```python
lis = [1,2,3,4,5]
np.random.shuffle(lis)
print(lis)
#[4, 5, 3, 1, 2]

#需要注意的是shuffle是作用于列表，所以，列表是动态修改的，如果直接打印更改的，会是"None"，需要直接打印列表本身
```

*19.求平均值*
```python
a = np.array([[1,2,3],[4,5,6]])
print(np.mean(a))
#3.5
print(np.mean(a,axis = 0))
#[2.5 3.5 4.5]
print(np.mean(a,axis = 1))
#[2. 5.]
print(np.mean(a, axis=0, keepdims=True))
#[[2.5 3.5 4.5]]
print(np.mean(a, axis=0, keepdims=True,dtype = int))
#[[2 3 4]]

#所有参数用法均与np.sum一样,不多赘述

```
*20.拓展维度*

```python
a = np.array([1,2,3])
b = np.expand_dims(a,axis = 0)
c = np.expand_dims(b,axis = 1)
d = np.expand_dims(b,-1)
print(a.shape)
print(b.shape)
print(c.shape)
print(d.shape)

------
(3,)
(1,3)
(1,1,3)
(1,3,1)
------
#axis = 0 最左边加一个维度
#axis = 1 中间加一个维度
#axis = -1 最右边加一个维度
```