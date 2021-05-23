
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

输入之后，紧接着进行词嵌入处理，词嵌入就是将每一个词用预先训练好的向量进行映射，参数为词表的大小和被映射的向量的维度，通俗来说就是向量里面有多少个数。注意，第一个参数是词表的大小，如果你目前有4个词，就填4，你后面万一进入与这4个词不同的词，还需要重新映射，为了统一，一开始的也要重新映射，因此这里填词表总大小。假如我们打算映射到512维（num_features或者embed_dim），那么，整个文本的形状变为100 x 128 x 512。接下来举个小例子解释一下：假设我们词表一共有10个词，文本里有2个句子，每个句子有4个词，我们想要把每个词映射到8维的向量。于是2，4，8对应于batch_size, seq_length, embed_dim（本文将batch放在第一维）。

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

多头注意力大概分为三个部分讲，分别为query，key，value初始化，注意力mask，点积注意力

- 初始化参数

query，key，value是源语言序列（本文记为src）乘以对应的矩阵得到的，那么，那些矩阵从何而来（注意，因为大部分代码都是从源码中抽离出来的，因而常带有self等，最后会呈现组成好的，而行文过程中不会将整个结构呈现出来）：

```python
from torch.nn.parameter import Parameter
factory_kwargs = {'device': device, 'dtype': dtype}
if self._qkv_same_embed_dim is False:
    # 初始化前后形状维持不变
    # (seq_length x embed_dim) x (embed_dim x embed_dim) ==> (seq_length x embed_dim)
    self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
    self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
    self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
    self.register_parameter('in_proj_weight', None)
else:
    self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
    self.register_parameter('q_proj_weight', None)
    self.register_parameter('k_proj_weight', None)
    self.register_parameter('v_proj_weight', None)

if bias:
    self.in_proj_bias = Parameter(torch.empty(3 * embed_dim), **factory_kwargs)
else:
    self.register_parameter('in_proj_bias', None)
# 后期会将所有头的注意力拼接在一起然后乘上权重矩阵输出
# out_proj是为了后期准备的
self.out_proj = Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
self._reset_parameters()
```
torch.empty是按照所给的形状形成对应的tensor，特点是填充的值还未初始化，类比torch.randn（标准正态分布），这就是一种初始化的方式。在PyTorch中，变量类型是tensor的话是无法修改值的，而Parameter()函数可以看作为一种类型转变函数，将不可改值的tensor转换为可训练可修改的模型参数，即与model.parameters绑定在一起，register_parameter的意思是是否将这个参数放到model.parameters，None的意思是没有这个参数。每个参数其实还有device和dtype两个属性，因此**factory_kwargs的意思是这两个参数是可变的。

这里有个if判断，用以判断q,k,v的最后一维是否一致，若一致，则一个大的权重矩阵全部乘然后分割出来，若不是，则各初始化各的，其实初始化是不会改变原来的形状的（如![](http://latex.codecogs.com/svg.latex?q=qW_q+b_q)，见注释）。

可以发现最后有一个_reset_parameters()函数，这个是用来初始化参数数值的。xavier_uniform意思是从[连续型均匀分布](https://zh.wikipedia.org/wiki/%E9%80%A3%E7%BA%8C%E5%9E%8B%E5%9D%87%E5%8B%BB%E5%88%86%E5%B8%83)里面随机取样出值来作为初始化的值，xavier_normal_取样的分布是正态分布。正因为初始化值在训练神经网络的时候很重要，所以才需要这两个函数。

constant_意思是用所给值来填充输入的向量。

另外，在PyTorch的源码里，似乎projection代表是一种线性变换的意思，in_proj_bias的意思就是一开始的线性变换的偏置

```python
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_

def _reset_parameters(self):
    if self._qkv_same_embed_dim:
        xavier_uniform_(self.in_proj_weight)
    else:
        xavier_uniform_(self.q_proj_weight)
        xavier_uniform_(self.k_proj_weight)
        xavier_uniform_(self.v_proj_weight)
    if self.in_proj_bias is not None:
        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)

```
以上便是参数初始化过程，接下来是进行query,key和value的赋值

***
- q,k,v从何来？



-  点积注意力

```python
from typing import Optional, Tuple
def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    r'''
    在query, key, value上计算点积注意力，若有注意力遮盖则使用，并且应用一个概率为dropout_p的dropout

    参数：
        - q: shape:`(B, Nt, E)` B代表batch size， Nt是目标语言序列长度，E是嵌入后的特征维度
        - key: shape:`(B, Ns, E)` Ns是源语言序列长度
        - value: shape:`(B, Ns, E)`与key形状一样
        - attn_mask: 要么是3D的tensor，形状为:`(B, Nt, Ns)`或者2D的tensor，形状如:`(Nt, Ns)`

        - Output: attention values: shape:`(B, Nt, E)`，与q的形状一致;attention weights: shape:`(B, Nt, Ns)`
    
    例子：
        >>> q = torch.randn((2,3,6))
        >>> k = torch.randn((2,4,6))
        >>> v = torch.randn((2,4,6))
        >>> out = scaled_dot_product_attention(q, k, v)
        >>> out[0].shape, out[1].shape
        >>> torch.Size([2, 3, 6]) torch.Size([2, 3, 4])
    '''
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2,-1))
    if attn_mask is not None:
        attn += attn_mask 
    # attn意味着目标序列的每个词对源语言序列做注意力
    attn = nn.functional.softmax(attn, dim=-1)
    if dropout_p:
        attn = nn.functional.dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn 
```

https://www.jianshu.com/p/d8b77cc02410