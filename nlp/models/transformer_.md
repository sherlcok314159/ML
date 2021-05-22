
### Transformer源代码解释之PyTorch篇

**章节**

- [词嵌入](#embed)
- [位置编码](#pos)
- [多头注意力](#multihead)
- [残差相连](#add&norm)
- [总结](#conclusions)
- [参考文献](#references)

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/transformer.png)

**<div id='embed'>词嵌入</div>**

Transformer本质上是一种Encoder，以翻译任务为例，原始数据集是以两种语言组成一行的，在应用时，应是Encoder输入源语言序列，Decoder里面输入需要被转换的语言序列（训练时）。一个文本常有许多序列组成，常见操作为将序列进行一些预处理（如词切分等）变成列表，一个序列的列表的元素通常为词表中不可切分的最小词，整个文本就是一个大列表，元素为一个一个由序列组成的列表。如一个序列经过切分后变为["am", "##ro", "##zi", "accused", "his", "father"]，接下来按照它们在词表中对应的索引进行转换，假设结果如[23, 94, 13, 41, 27, 96]。假如整个文本一共100个句子，那么就有100个列表为它的元素，因为每个序列的长度不一，需要设定最大长度，这里不妨设为128，那么将整个文本转换为数组之后，形状即为100 x 128，这就对应着batch_size和seq_length。

输入之后，紧接着进行词嵌入处理，词嵌入就是将每一个词用预先训练好的向量进行映射，参数为词表的大小和被映射的向量的维度，通俗来说就是向量里面有多少个数。注意，第一个参数是词表的大小，如果你目前有4个词，就填4，你后面万一进入与这4个词不同的词，还需要重新映射，为了统一，一开始的也要重新映射，因此这里填词表总大小。假如我们打算映射到512维（num_features或者num_hiddens），那么，整个文本的形状变为100 x 128 x 512。接下来举个小例子解释一下：假设我们词表一共有10个词，文本里有2个句子，每个句子有4个词，我们想要把每个词映射到8维的向量。于是2，4，8对应于batch_size, seq_length, num_features（本文将batch放在第一维）。

另外，一般深度学习任务只改变num_features，所以讲维度一般是针对最后特征所在的维度。

```python
import torch
import torch.nn as nn
X = torch.zeros((2,4),dtype=torch.long)
embed = nn.Embedding(10,8)
print(embed(X).shape)
# torch.Size([2, 4, 8])
```

***

**<div id='pos'>位置编码</div>**

词嵌入之后紧接着就是位置编码，位置编码用以区分不同词以及同词不同特征之间的关系。代码中需要注意：X_只是初始化的矩阵，并不是输入进来的；完成位置编码之后会加一个dropout。另外，位置编码是最后加上去的，因此输入输出形状不变。

```python
Tensor = torch.Tensor
def positional_encoding(X, num_features, dropout_p=0.0, max_len=512) -> Tensor:
    r'''
        给输入加入位置编码
    参数：
        - num_features: 输入进来的维度
        - dropout_p: dropout的概率，当其为非零元素时执行dropout
        - max_len: 句子的最大长度，默认512
    
    形状：
        - 输入： [batch_size, seq_length, num_features]
        - 输出： [batch_size, seq_length, num_features]

    例子：
        >>> X = torch.randn((2,4,10))
        >>> X = positional_encoding(X, 10)
        >>> print(X.shape)
        >>> torch.Size([2, 4, 10])
    '''

    dropout = nn.Dropout(dropout_p)
    P = torch.zeros((1,max_len,num_features))
    X_ = torch.arange(max_len,dtype=torch.float32).reshape(-1,1) / torch.pow(
        10000,
        torch.arange(0,num_features,2,dtype=torch.float32) /num_features)
    P[:,:0::2] = torch.sin(X_)
    P[:,:,1::2] = torch.cos(X_)
    X = X + P[:,:X.shape[1],:].to(X.device)
    return dropout(X)
```
***

**<div id='multihead'>多头注意力</div>**

多头注意力分为大概三个部分讲，点积注意力，初始化参数，以及遮挡机制

-  点积注意力