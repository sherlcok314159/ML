### Sentiment Classification in IMDB

***通过DNN实现影评情感分类***

章节

- [包的准备](#prepare)
- [数据下载](#download)
- [数据预处理](#preprocess)
- [设计DNN](#design)
- [调参](#update)


**<div id='prepare'>包的准备</div>**

```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

for module in mpl,np,pd,tf,keras:
    print(module.__name__,module.__version__)

'''
matplotlib 3.3.4
numpy 1.16.0
pandas 1.1.5
tensorflow 1.14.0
tensorflow.python.keras.api._v1.keras 2.2.4-tf
'''

若没有按照版本安装，如
pip3 install numpy==1.16.0
```

**<div id='download'>数据下载</div>**

```python
#取出词频为前10000
vocab_size = 10000
# <3的id都是特殊字符
index_from = 3
# 这里用keras里面的imdb数据集
imdb = keras.datasets.imdb

(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words = vocab_size,index_from = index_from)

'''
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz
17465344/17464789 [==============================] - 28s 2us/step
'''

# 若要进一步了解各种参数，API示例文档 https://keras.io/api/datasets/imdb/

# 训练集的大小
print(train_data.shape)
print(train_labels.shape)

# (25000,)
# (25000,)

# 训练集的第一个样本（是向量）
print(train_data[0],train_labels[0])

'''
[1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4,……] 

1
'''

# 多维的数组 numpy.ndarray
print(type(train_data))
print(type(train_labels))

'''
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
'''

# train_labels的值0(negative),1(positive)-二分类
print(np.unique(train_labels))

# [0 1]

#测试集大小
print(test_data.shape)
print(test_labels.shape)

#(25000, 500)
#(25000,)

```

**<div id='preprocess'>数据预处理</div>**

```python
# 下载词表，就是imdb_word_index.json
# key是单词，value是索引
word_index=imdb.get_word_index()

print(len(word_index))
print(type(word_index))

# 88584
# <class 'dict'>

print(word_index.get("footwork"))

# 34698

# 取出的词表索引从1开始，id都偏移3
word_index = {k:(v+3) for k,v in word_index.items()}
# 自定义索引0-3
word_index["<PAD>"] = 0 # 填充字符
word_index["<START>"] = 1 # 起始字符
word_index["<UNK>"] = 2 # 找不到就返回UNK
word_index["<END>"] = 3 # 结束字符

# 转成习惯的方式，索引为key,单词为value
reverse_word_index = {v:k for k,v in word_index.items()}

# 查看解码效果
print(reverse_word_index)

# {34710: 'fawn', 52015: 'tsukino', 52016: 'nunnery', 16825: 'sonja', 63960: 'vani', 1417: 'woods', ……}

# 解码函数，如果找不到的话就是<UNK>，按照索引找词
def decode_review(text_ids):
  return " ".join([reverse_word_index.get(word_id,"<UNK>") for word_id in text_ids])


# 逐个对单词解码，得到一篇文本
decode_review(train_data[0])

# debug

print(reverse_word_index[34707])

# footwork

print(word_index.get("footwork"))

# 34707

# 随机看一下样本长度，可见长度不一
print(len(train_data[0]),len(train_data[1]),len(train_data[100]))

# 218 189 158

# 设置输入词汇表的长度，长度<500会被补全，>500会被截断

max_length = 500

# 填充padding
# value 用什么值填充
# padding 选择填充的顺序，2中pre,post
train_data = keras.prepocessing.sequence.pad_sequences(train_data,value = word_index["<PAD>"],padding="pre",maxlen = max_length)

# 使测试集要和训练集结构相同
test_data = keras.preprocessing.sequence.pad_sequences(test_data,value = word_index["<PAD>"],padding = "pre",maxlen = max_length)

```


这里简要介绍一下两种**padding**方式，与前面代码无关，一个是在**开头**填充，一个是在**末尾**填充，在本题中好像pre train出的结果好一点

```python

train_data = [[1,2,3,34],[2,3,1,4,2]]
train_data = keras.preprocessing.sequence.pad_sequences(train_data,value = 0,padding = "pre",maxlen = 10)
print(train_data)

'''
[[ 0  0  0  0  0  0  1  2  3 34]
 [ 0  0  0  0  0  2  3  1  4  2]]
'''

train_data = [[1,2,3,34],[2,3,1,4,2]]
train_data = keras.preprocessing.sequence.pad_sequences(train_data,value = 0,padding = "post",maxlen = 10)

print(train_data)

'''
[[ 1  2  3 34  0  0  0  0  0  0]
 [ 2  3  1  4  2  0  0  0  0  0]]
'''
```

**<div id='design'>定义模型</div>**

```python
# 一个单词的维度是16维
embedding_dim = 16
batch_size = 128

# 定义模型
# 定义矩阵 [vocab_size,embedding_dim]
# GlobalAveragePooling1D 全局平均值池化-在max_length这个维度上做平均，就是1x16了,在哪个维度上做Global，该维度就会消失
# 二分类问题，最后的激活函数用sigmoid
model = keras.models.Sequential([
        keras.layers.Embedding(vocab_size,embedding_dim,input_length = max_length),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(64,activation = "relu"),
        keras.layers.Dense(1,activation = "sigmoid"),
])

model.summary()
model.compile(optimizer = "adam",loss = "binary_crossentropy",metrics = ["accuracy"])

# 数据集中只有训练集、测试集，没有验证集，就用validation_split-拿20%的训练集数据当作验证集数据
history=model.fit(train_data,train_labels,epochs=30,batch_size=batch_size,validation_split=0.2)

# 绘制学习曲线
def plot_(history,label):
    plt.plot(history.history[label])
    plt.plot(history.history["val_" + label])
    plt.title("model " + label)
    plt.ylabel(label)
    plt.xlabel("epoch")
    plt.legend(["train","validation"],loc = "upper left")
    plt.show()

plot_(history,"accuracy")
plot_(history,"loss")

```

![](https://github.com/sherlcok314159/ML/blob/main/Images/imdb_1.png)

![](https://github.com/sherlcok314159/ML/blob/main/Images/imdb_2.png)

其实可以发现已经烂掉了，发生了过拟合

```python
score = model.evaluate(
    test_data,test_labels,
    batch_size=batch_size,
    verbose=1
)

print("Test loss: ", score[0])
print("Test accuracy: ", score[1])

'''
Test loss: 0.7127751180744171
Test accuracy: 0.85788
'''
```

接下来进行一些讨论

**<div id='update'>调参</div>**

**More neurons?**

```python
# 改成128个神经元
keras.layers.Dense(128,activation='relu'),
'''
Test loss: 0.7896
Test accuracy: 0.8556
'''

# 改成256个神经元
keras.layers.Dense(256,activation='relu'),
'''
Test loss: 0.9015
Test accuracy: 0.8531
'''

# 改成512个神经元
keras.layers.Dense(512,activation='relu'),
'''
Test loss: 1.0439
Test accuracy: 0.8510
'''
# 其实结果都反倒不好了
```

**Other activation functions?**
```python
# 试试看sigmoid
keras.layers.Dense(64,activation='sigmoid'),
'''
Test loss: 0.3955
Test accuracy: 0.8771
'''
# 哇，这个结果很棒！

# 试试看tanh
keras.layers.Dense(64,activation='tanh'),
'''
Test loss: 1.0261
Test accuracy: 0.8558
'''
# 没什么区别吧

# 试试看softmax
keras.layers.Dense(64,activation='softmax'),                               
'''
Test loss: 0.2950
Test accuracy: 0.8838
'''

# Cool!

# 试试看softplus
keras.layers.Dense(64,activation='softplus'),                               
'''
Test loss: 0.4718
Test accuracy: 0.8709
'''
# 还不错

# 试试看Elu
keras.layers.Dense(64,activation='elu'),                               
'''
Test loss: 0.9268
Test accuracy: 0.8544
'''

# 试试看Softsign
keras.layers.Dense(64,activation='softsign'),                               
'''
Test loss: 0.9815
Test accuracy: 0.8555
'''

# 试试看Exponential
keras.layers.Dense(64,activation='exponential'),                               
'''
Test loss: 0.9815
Test accuracy: 0.8555
'''


# 试试看Leaky Relu
keras.layers.Dense(64),
keras.layers.LeakyRelu(),                               
'''
Test loss: 0.8014
Test accuracy: 0.8560
'''
```

**Deep?**

尝试了各种各样的组合，最终大部分都落在[86，87]区间，还没有不DEEP来的好


**EarlyStopping?**

其实很多时候train的时候再往下其实val_loss是不会下降的，不如直接停掉，防止过拟合

代码在[Problems](../problems.md)中说过了，这里只讨论一下patience参数的影响


```python
patience = 0，accuracy: 0.8788
```
![](https://github.com/sherlcok314159/ML/blob/main/Images/val_loss_0.png)

```python
patience = 1，accuracy: 0.8746
```
![](https://github.com/sherlcok314159/ML/blob/main/Images/val_loss_1.png)

```python
patience = 2，accuracy: 0.8772
```

![](https://github.com/sherlcok314159/ML/blob/main/Images/val_loss_2.png)


```python
patience = 3，accuracy: 0.8183
```

![](https://github.com/sherlcok314159/ML/blob/main/Images/val_loss_3.png)

```python
patience = 4，accuracy: 0.8724
```

![](https://github.com/sherlcok314159/ML/blob/main/Images/val_loss_4.png)

```python
patience = 4，accuracy: 0.8650
```

![](https://github.com/sherlcok314159/ML/blob/main/Images/val_loss_5.png)

其实大概可以看出，很多时候我们很难找出一个patience充当万金油，不知道该跳哪一个，所以真正实操的时候，很多时候选择不跳过

一般的取值在10到100之间，取决于你的data和模型。举个例子来讲，如果你的模型下降的比较慢一开始，就把patience适当调大一点

