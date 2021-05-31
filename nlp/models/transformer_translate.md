### Transformer之翻译篇

章节

- [Pipeline构建](#preprocess)
- [DataLoader构建](#build)
- [MASK机制](#mask)
- [模型搭建](#model)
- [自定义学习率](#special)
- [train和validation](#mix)
- [模型训练](#train)
- [评估](#eval)
- [讨论](#discuss)
    - 卷积代替全连接？
    - Bert作为Encoder?
    - 注意力头剪枝?
    - epoch越多越好？
- [参考文献](#references)

源码在[colab](https://colab.research.google.com/drive/1CILp7vwm8bZy6dOnRuwPeujP3-Mdm67z?usp=sharing)上，数据集若要自己下载[data](../RNN/eng-fra.txt)

### <div id='preprocess'>Pipeline构建</div>

必备包的导入

```python
import torch 
import torchtext

from sklearn.model_selection import train_test_split

import random
import re 
from tqdm import tqdm
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt 
import unicodedata
import datetime
import time
from torchtext.legacy.data import Field, Dataset, Example, Iterator
import copy
import torch.nn as nn 
import matplotlib.pyplot as plt
import os 
import pdb # jupyter调试用，一般不需要

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device=",device)
```

首先用pandas读入：

```python
data_df = pd.read_csv("./eng-fra.txt",encoding="utf-8",sep="\t",header=None,names=["eng","fra"],index_col=False)

print(data_df.shape)
print(data_df.values.shape)
print(data_df.values[0])
print(data_df.values[0].shape)
data_df.head()

# (135842, 2)
# (135842, 2)
# ['Go.' 'Va !']
# (2,)
'''
	eng	fra
0	Go.	Va !
1	Run!	Cours !
2	Run!	Courez !
3	Wow!	Ça alors !
4	Fire!	Au feu !
'''
```
详见注释

```python
# 数据预处理
# 将unicode字符串转化为ASCII码：
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


# 规范化字符串
def normalizeString(s):
    # print(s) # list  ['Go.']
    # s = s[0]
    s = s.lower().strip()
    s = unicodeToAscii(s)
    s = re.sub(r"([.!?])", r" \1", s)  # \1表示group(1)即第一个匹配到的 即匹配到'.'或者'!'或者'?'后，一律替换成'空格.'或者'空格!'或者'空格？'
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)  # 非字母以及非.!?的其他任何字符 一律被替换成空格
    s = re.sub(r'[\s]+', " ", s)  # 将出现的多个空格，都使用一个空格代替。例如：w='abc  1   23  1' 处理后：w='abc 1 23 1'
    return s
```

```python
print(normalizeString('Va !'))
print(normalizeString('Go.'))

# va !
# go .
```


```python
MAX_LENGTH = 10

eng_prefixes = (  # 之前normalizeString()已经对撇号等进行了过滤，以及清洗，小写化等
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

# print(eng_prefixes)
pairs = [[normalizeString(s) for s in line] for line in data_df.values]
print('pairs num=', len(pairs))
print(pairs[0])
print(pairs[1])

# pairs num= 135842
# ['go .', 'va !']
# ['run !', 'cours !']
```

```python
# 文件是英译法，我们实现的是法译英，所以进行了reverse，所以pair[1]是英语
# 为了快速训练，仅保留“我是”“你是”“他是”等简单句子，并且删除原始文本长度大于10个标记的样本
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and \
           p[0].startswith(eng_prefixes)  # startswith first arg must be str or a tuple of str


def filterPairs(pairs):
    # 过滤，并交换句子顺序，得到法英句子对（之前是英法句子对）
    return [[pair[1], pair[0]] for pair in pairs if filterPair(pair)]


pairs = filterPairs(pairs)
```

```python
print('after trimming, pairs num=', len(pairs))
print(pairs[0])
print(pairs[1])
print(random.choice(pairs))
print(pairs[0:2])

# after trimming, pairs num= 10599
# ['j ai ans .', 'i m .']
# ['je vais bien .', 'i m ok .']
# ['on nous fait chanter .', 'we re being blackmailed .']
# [['j ai ans .', 'i m .'], ['je vais bien .', 'i m ok .']]
```
以上代码的目的是首先对特殊字符进行修正`（i'm ==> i m)`并将字符与字母之间留出空格，其次按照序列长度和特定开头对数据进行筛选以便后期的训练，最终将`pairs`变成大列表，每一个元素即为小列表，小列表第一，二个元素分别为法文和英文序列。

```python
# 划分数据集： 训练集和验证集
train_pairs, val_pairs = train_test_split(pairs, test_size=0.2, random_state=1234)

print(len(train_pairs))
print(len(val_pairs))

# 8479
# 2120
```

***

### <div id='build'>DataLoader构建</div>

第一部分我们构建了数据基础处理的流水线，接下来构建可以迭代并且符合PyTorch数据格式的DataLoader。

不同的Field其实是针对不同的数据的，`SRC_TEXT`和`TGT_TEXT`分别针对法语和英语（本次的任务是法译英）

```python
tokenizer = lambda x: x.split()

SRC_TEXT = Field(sequential=True,
tokenize=tokenizer,
# +2 因为<start>和<end>
fix_length=MAX_LENGTH+2,
preprocessing=lambda x:["<start>"] + x + ["<end>"])

TGT_TEXT = Field(sequential=True,
tokenize=tokenizer,
fix_length=MAX_LENGTH+2,
preprocessing=lambda x:["<start>"] + x + ["<end>"])

def get_dataset(pairs, src, tgt):
    # field信息：fields dict([str, Field])
    fields = [("src",src),("tgt",tgt)]
    # list(Example)
    examples = [] 
    # tqdm用以计时
    for fra, eng in tqdm(pairs):
        # 创建Example时会调用field.preprocessing方法
        examples.append(Example.fromlist([fra,eng],fields))
    return examples, fields
```

准备训练和测试数据集

```python
ds_train = Dataset(*get_dataset(train_pairs, SRC_TEXT, TGT_TEXT))
ds_val = Dataset(*get_dataset(val_pairs, SRC_TEXT, TGT_TEXT))

print(len(ds_train[0].src), ds_train[0].src)
print(len(ds_train[0].tgt), ds_train[0].tgt)

# 9 ['<start>', 'tu', 'n', 'es', 'qu', 'un', 'lache', '.', '<end>']
# 8 ['<start>', 'you', 're', 'just', 'a', 'coward', '.', '<end>']
```

构建法语词表，这里务必注意`<pad>`是1而非0，`<unk>`表示unknown，即为词表中未出现过的词

```python
SRC_TEXT.build_vocab(ds_train)
print(len(SRC_TEXT.vocab))
print(SRC_TEXT.vocab.itos[0])
print(SRC_TEXT.vocab.itos[1])
print(SRC_TEXT.vocab.itos[2])
print(SRC_TEXT.vocab.itos[3])
print(SRC_TEXT.vocab.stoi["<start>"])
print(SRC_TEXT.vocab.stoi["<end>"])

# 3901
# <unk>
# <pad>
# <end>
# <start>
# 3
# 2
```

模拟decode

```python
res = []
for id in [3, 5, 6, 71, 48, 5, 8, 32, 743, 4, 2, 1]:
    res.append(SRC_TEXT.vocab.itos[id])
print(" ".join(res)+"\n")
# <start> je suis fais si je vous l examen . <end> <pad>
```
构建英文词表，关于`<unk>`,`<pad>`均与法语词表一致

```python
TGT_TEXT.build_vocab(ds_train)
print(len(TGT_TEXT.vocab))
# 2591
```

```python
BATCH_SIZE = 64

# 构建数据管道迭代器
# split可以分开处理训练和验证集
# train_iter由许多batch组成
train_iter, val_iter = Iterator.splits(
    (ds_train, ds_val),
    # batch内部对数据是否排序
    # 上面有ds_train[0].src
    # 根据每一条数据src的长度进行降序排列
    sort_within_batch=True,
    sort_key=lambda x:len(x.src),
    batch_sizes=(BATCH_SIZE, BATCH_SIZE)
)

# 查看数据管道信息，会触发postprocessing，如果有的话
for batch in train_iter:
    # 注意，这里text第一维是seq_len，而非batch
    print(batch.src[:,0])
    print(batch.src.shape, batch.tgt.shape)
    break
```

```python
tensor([3, 29, 17, 33, 82, 31, 381, 363, 3591, 4, 2, 1])
torch.Size([12, 64]) torch.Size([12, 64])
```

构建DataLoader
```python
class DataLoader:
    def __init__(self, data_iter):
        self.data_iter = data_iter
        self.length = len(data_iter)
    
    def __len__(self):
        return self.length
    
    def __iter__(self):
        # 注意，此处调整text的shape为batch_first
        for batch in self.data_iter:
            yield(torch.transpose(batch.src, 0, 1), torch.transpose(batch.tgt, 0, 1))

train_dataloader = DataLoader(train_iter)
val_dataloader = DataLoader(val_iter)
```

```python
# 查看数据管道
print("len(train_dataloader):",len(train_dataloader))
for batch_src, batch_tgt in train_dataloader:
    print(batch_src.shape, batch_tgt.shape)
    print(batch_src[0], batch_src.dtype)
    print(batch_tgt[0], batch_tgt.dtype)
    break

# len(train_dataloader): 133
# torch.Size([64, 12]) torch.Size([64, 12])
# tensor([  3,  13,  15,  21,   9,  17,  46,  10, 230,   4,   2,   1]) torch.int64
# tensor([  3,  14,   6,  10, 210,   4,   2,   1,   1,   1,   1,   1]) torch.int64

```
***
### <div id='mask'>MASK机制</div>

原始的句子首先需要转换为词表中的索引，然后进入词嵌入层。举个例子，假如某个时间步长上输入句子为`"I love u"`，src_vocab（源语言词表）为`{"SOS":0,"EOS":1,"I":2,"love":3,"u":4}`，`SOS`和`EOS`代表句子的开头和末尾，那么输入句子变为`[[2, 3, 4, 1]]`，接下来进入词嵌入层，目前我们的词表只有5个词，所以`embed = nn.Embedding(5, 6)`，用一个6维向量表示每一个词，如下所示：

```python
import torch
import torch.nn as nn
t = torch.tensor([[2, 3, 4, 1]])
embed = nn.Embedding(5, 6)
print(embed(t))
'''
tensor([[[-0.5557,  0.9911, -0.2482,  1.5019,  0.9141,  0.0697],
         [-1.5058, -0.4237,  1.1189, -0.7472, -0.9834, -1.2829],
         [-0.8558,  0.4753,  0.0555, -0.3921, -0.0232,  0.2518],
         [-0.3563,  0.7707, -0.8797,  0.6719, -0.4903,  0.0508]]],
       grad_fn=<EmbeddingBackward>)
'''       
```

- PAD MASK

但现实中往往还要复杂的多，其中最重要的问题即为不定长，在RNN中序列可以不定长，可是在transformer中是需要定长的，这意味着需要设置一个最大长度，小于最大长度的全部补0，这些补0的就被称为`<PAD>`，对应的索引常为0，src_vocab扩充为`{"PAD":0,"SOS":1,"EOS":2,"I":3,"love":4,"u":5}`，假如最大长度为6，那么输入句子对应的索引变为`[[2, 3, 4, 1, 0, 0]]`。长度统一之后，新的问题产生了：那些补0的地方机器并不知道，在做注意力运算的时候还是要加入运算，这样结果带有一定的误导性，为了减免填充的地方对注意力运算带来影响，在实际运用中用`MASK`遮住补0的地方。

```python
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # 等于0的即为<PAD>
    # .data意思是不在计算图中储存它的梯度
    # eq意思是equal，是否相等
    pad_attn_mask = seq_k.data.eq(0).unsequeeze(1)
    # pad_attn_mask ==> [batch_size, 1, seq_k_length]
    return pad_attn_mask.expand(batch_size, len_q, len_k)
```

为什么需要让pad_attn_mask的形状为`(batch_size, len_q, len_k)`呢？众所周知，做注意力的时候是query去与key做点积运算，做embedding之后q和k的形状为`(batch_size, q_len, embed_size)`和`(batch_size, k_len, embed_size)`，于是两者做点积后的shape变为`(batch_size, q_len, k_len)`，`MASK`需要与`attn_mask`形状一致。

`pad_attn_mask`是针对`key`的，`True`的地方代表是填充的，做自注意力的时候，q和k都是q，接下来实例看一下：

```python
q = torch.tensor([[2, 3, 4, 1, 0, 0]])
mask = get_attn_pad_mask(q, q) 
print(mask)
'''
tensor([[[False, False, False, False,  True,  True],
         [False, False, False, False,  True,  True],
         [False, False, False, False,  True,  True],
         [False, False, False, False,  True,  True],
         [False, False, False, False,  True,  True],
         [False, False, False, False,  True,  True]]])
'''
```
这里有一个容易混淆的点，`get_attn_pad_mask`里的参数只是需要索引转成tensor之后的就可以了，不必经过Embedding层，因为我们获得注意力mask只是想将索引为0的部分给填充掉，况且若经过Embedding之后也找不到0了。而做注意力点积的时候，是需要经过Embedding之后的。

观察下面的示例，不难发现mask里面为`True`的全被填上了`-inf`，这样经过softmax之后值即为0，是不会对结果造成影响的。

attn_mask有两种主流操作，一是先不管，正常进行，在softmax之前masked_fill将`1`和`True`的地方填充为`-inf`；二是一开始先将mask的组成变为`0`和`-inf`，然后直接加上去，加0的地方自然无影响，加`-inf`地方进行softmax之后全为0。

下面为最后填充的做法

```python
embed = nn.Embedding(5, 6)
q = embed(q)
attn = torch.bmm(q, q)
attn = attn.masked_fill(get_attn_pad_mask, float("-inf"))
print(attn)
'''
tensor([[[ 0.8943, -1.5924,  0.3291,  2.9090,    -inf,    -inf],
         [ 7.0428,  3.5899, 14.1080,  1.9872,    -inf,    -inf],
         [ 4.8335,  1.7810,  6.7327,  2.5438,    -inf,    -inf],
         [ 0.9912,  0.1271,  1.1755,  5.6559,    -inf,    -inf],
         [-2.8715, -1.5905, -6.0587,  5.0048,    -inf,    -inf],
         [-2.8715, -1.5905, -6.0587,  5.0048,    -inf,    -inf]]],
       grad_fn=<MaskedFillBackward0>)
'''
softmax = nn.Softmax(dim=-1)
attn = softmax(attn)
print(attn)
'''
tensor([[[1.0929e-01, 9.0916e-03, 6.2105e-02, 8.1951e-01, 0.0000e+00,
          0.0000e+00],
         [8.5359e-04, 2.7020e-05, 9.9911e-01, 5.4402e-06, 0.0000e+00,
          0.0000e+00],
         [1.2773e-01, 6.0340e-03, 8.5329e-01, 1.2938e-02, 0.0000e+00,
          0.0000e+00],
         [9.1945e-03, 3.8746e-03, 1.1054e-02, 9.7588e-01, 0.0000e+00,
          0.0000e+00],
         [3.7898e-04, 1.3644e-03, 1.5646e-05, 9.9824e-01, 0.0000e+00,
          0.0000e+00],
         [3.7898e-04, 1.3644e-03, 1.5646e-05, 9.9824e-01, 0.0000e+00,
          0.0000e+00]]], grad_fn=<SoftmaxBackward>)
'''
```

当然也可用第二种方法，因为每次Embedding权重不一，执行了两种操作我分别用了不同的文件来执行，故结果有所不同。

```python
mask = mask.float().masked_fill(mask == 1, float("-inf")).masked_fill(mask == 0, float(0.0))
print(mask)
'''
tensor([[[0., 0., 0., 0., -inf, -inf],
         [0., 0., 0., 0., -inf, -inf],
         [0., 0., 0., 0., -inf, -inf],
         [0., 0., 0., 0., -inf, -inf],
         [0., 0., 0., 0., -inf, -inf],
         [0., 0., 0., 0., -inf, -inf]]])
'''
attn += mask
attn = softmax(attn)
print(attn)
'''
tensor([[[7.2725e-01, 1.4629e-03, 2.6949e-01, 1.7929e-03, 0.0000e+00,
          0.0000e+00],
         [1.4819e-01, 6.5444e-03, 1.0565e-05, 8.4526e-01, 0.0000e+00,
          0.0000e+00],
         [1.0830e-02, 7.3555e-01, 2.4050e-01, 1.3125e-02, 0.0000e+00,
          0.0000e+00],
         [1.0847e-02, 9.8162e-01, 6.8870e-03, 6.4141e-04, 0.0000e+00,
          0.0000e+00],
         [1.1121e-01, 7.8785e-01, 5.9839e-05, 1.0088e-01, 0.0000e+00,
          0.0000e+00],
         [1.1121e-01, 7.8785e-01, 5.9839e-05, 1.0088e-01, 0.0000e+00,
          0.0000e+00]]], grad_fn=<SoftmaxBackward>)
'''
```

- Position Mask

众所周知，在解码的时候，首先还是对decode_input做自注意处理，这个时候我们需要输入进一个词从而让机器决定下一个可能出现的词，若对decode_input不做处理的话，全部看到就犯规了，等于白训练。

即position mask的作用是让某位置只能关照它和它之前的位置，而不是之后的位置也可以看到。

```python
def generate_square_subsequent_mask(sz: int):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
```

不妨继续上面的例子，q依旧是`[[2, 3, 4, 1, 0, 0]](tensor形式)`，此时q变成了解码器里面的输入，不仅要遮挡住`<PAD>`，同时还得让每个位置只能瞻前不能顾后

```python
position_mask = generate_square_subsequent_mask(q.size[-1])
print(position_mask)
'''
tensor([[0., -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf, -inf],
        [0., 0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0., 0.]])
'''
# 整合PAD MASK和位置遮挡
attn_mask = mask + position_mask
attn += attn_mask
attn = softmax(attn)
print(attn)
'''
tensor([[[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.2119, 0.7881, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.1788, 0.6952, 0.1260, 0.0000, 0.0000, 0.0000],
         [0.1402, 0.2170, 0.1522, 0.4907, 0.0000, 0.0000],
         [0.1363, 0.0096, 0.0186, 0.8355, 0.0000, 0.0000],
         [0.1363, 0.0096, 0.0186, 0.8355, 0.0000, 0.0000]]],
       grad_fn=<SoftmaxBackward>)
'''
```

两种seq2seq模型必备的注意力机制到此为止，接下来讲一下transformer翻译模型的各个模块
***

### <div id='model'>模型搭建</div>

关于Transformer的[原理](transformer.md)以及[源码](transformer_.md)（PyTorch）并不是本文的重点，如有不了解的地方，按照相应的地方进行了解。

- Positional Encoding

这里我多返回了一个位置矩阵用以后面可视化

```python
Tensor = torch.Tensor
# 给一个tensor所有位置的位置编码
def positional_encoding(X, num_features, dropout_p=0.1, max_len=128) -> Tensor:
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
    P[:,:,0::2] = torch.sin(X_)
    P[:,:,1::2] = torch.cos(X_)
    X = X + P[:,:X.shape[1],:].to(X.device)
    return dropout(X), P
```
定义可视化注意力图函数
```python
def draw_pos_encoding(pos_encoding, d_model):
    plt.figure()
    plt.pcolormesh(pos_encoding[0],cmap="RdBu")
    plt.xlabel("Depth")
    plt.xlim((0, d_model))
    plt.ylabel("Position")
    plt.colorbar()
    plt.show()

X = torch.randn((2,4,128))
pos_encoding = positional_encoding(X, 128)
draw_pos_encoding(pos_encoding[1], 128)
```

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/pos.png)


- MASK 

上述的讲解以及大多数情况pad mask均为0，这里torchtext是1，尤其小心！

```python
# 在NLP中，<PAD>用以填充句子，而这没有携带任何信息，故需要被mask掉
# 返回True和False组成的mask
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(1) 即为<PAD>
    # 注意torchtext是<pad>为1而非0
    pad_attn_mask = seq_k.data.eq(1).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)

# 返回1，0组成的mask
def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = torch.triu(torch.ones(attn_shape),1)
    subsequent_mask = subsequent_mask.byte()
    return subsequent_mask
```

- ScaledDotProductAttention

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) 
        # 1的位置全部填充以忽略
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn
```

下面的矩阵初始化以及乘积可能有些迷惑，先看一个引例

```python
w = nn.Linear(128, 10 * 8)
q = torch.randn((10, 10, 128))
# 第一个10不动，(10 x 128) x (128 x 80) => 10 x 80
# 无转置，其实是q x w
w(q).shape
# torch.Size([10, 10, 80])
```


- MultiHeadAttention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        # Q: [batch_size x len_q x d_model], K: [batch_size x len_k x d_model], V: [batch_size x len_k x d_model]

        residual, batch_size = Q, Q.size(0)
        
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # q_s: [batch_size x n_heads x len_q x d_k]
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        
        # k_s: [batch_size x n_heads x len_k x d_k]  
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        
        # v_s: [batch_size x n_heads x len_k x d_v]  
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  
        
        # attn_mask : [batch_size x n_heads x len_q x len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        
        # context: [batch_size x n_heads x len_q x d_v]
        # attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        
        # context: [batch_size x len_q x n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) 
        
        # output: [batch_size x len_q x d_model]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn 
```


- PoswiseFeedForwardNet


```python
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, inputs):
        residual = inputs 
        output = self.dropout(nn.ReLU()(self.linear1(inputs)))
        output = self.layer_norm(self.dropout(self.linear2(output)) + residual)
        return output
```

- EncoderLayer

在Encode的时候我们只需要使用pad_mask即可，PyTorch的transformer库有很多mask，如src_mask，src_key_padding_mask，前者按照官方的意思是让某些位置被忽略，后者是与pad_mask等价的，在训练Encoder的时候，两者等价，只需要一个src_key_padding_mask即可，这里即为`enc_self_attn_mask`

```python
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) 
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn
```

- DecoderLayer

记得上面的dec_inputs是两种MASK结合一起的，即为`dec_self_attn_mask`，`dec_enc_self_mask`是针对enc_inputs来说的key_padding_mask，与上面的enc_self_attn_mask本质一致，只是shape不一样，一个是`[batch_size, enc_size, enc_size]`，另一个是`[batch_size, dec_size, enc_size]`。

dec_inputs首先经过解码器的自注意运算，然后充当query去查enc_outputs

```python
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn
```

- Encoder

注意其实源语言和目标语言不共用一个Embedding

```python
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.dropout = nn.Dropout(0.1)

    def forward(self, enc_inputs):
        enc_outputs = self.dropout(positional_encoding(self.src_emb(enc_inputs), d_model)[0])
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs).to(device)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns
```

- Decoder
```python
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        dec_outputs = positional_encoding(self.tgt_emb(dec_inputs), d_model)[0]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(device)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs).to(device)
        # torch.gt严格大于
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs).to(device)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, 
            dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns
```

- Transformer

```python
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs) 
        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns
```
***
### <div id='special'>自定义学习率</div>

按照论文里面的，我们还需要自定义学习率，不难发现一开始学习率上升比较快，后期学习率缓慢下降，最终趋于平衡

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/math.png)

```python
class CustomSchedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warm_steps=4):
        self.optimizer = optimizer
        self.warmup_steps = warm_steps

        super(CustomSchedule, self).__init__(optimizer)

    def get_lr(self):
        arg1 = self._step_count ** (-0.5)
        arg2 = self._step_count * (self.warmup_steps ** (-1.5))
        dynamic_lr = (d_model ** (-0.5)) * min(arg1, arg2)
        return [dynamic_lr for group in self.optimizer.param_groups]
```

接下来可视化看一下学习率的变化情况

```python
# 测试
import warnings
warnings.filterwarnings("ignore")

n_layers, d_model, n_heads, d_ff  = 6, 512, 8, 2048

src_vocab_size = len(SRC_TEXT.vocab) 
tgt_vocab_size = len(TGT_TEXT.vocab) 

d_k = d_v = d_model // n_heads

model = Transformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
learning_rate = CustomSchedule(optimizer, warm_steps=4000)


lr_list = []
for i in range(1, 20000):
    learning_rate.step()
    lr_list.append(learning_rate.get_lr()[0])
plt.figure()
plt.plot(np.arange(1, 20000), lr_list)
plt.legend(['warmup=4000 steps'])
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")
plt.show()
```

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/lr.png)

***
### <div id='mix'>train和validation</div>

先是train_step，详见注释

```python
def train_step(inp,tgt,criterion):
    # 采取teacher_forcing，将上一个时间步的标签直接输入模型
    tgt_input = tgt[:,:-1]
    tgt_real = tgt[:,1:]

    # to(device)不能少，否则默认cpu
    inp = inp.to(device)
    tgt_input = tgt_input.to(device)
    tgt_real = tgt_real.to(device)

    model.train()
    optimizer.zero_grad()

    # pred ==> [bz, seq_len, vocab_size]
    # tgt_real ==> [bz, seq_len]
    pred, enc_self_attns, dec_self_attns, dec_enc_attns = model(inp, tgt_input)
    # pdb.set_trace()
    # transformed_pred ==> [bz x seq_len, vocab_size]
    # transformed_tgt_real ==> [bz x seq_len, ]
    loss = criterion(pred.view(-1, pred.size(-1)), tgt_real.contiguous().view(-1))
    # sum()不能忘，item()只能转换标量值为python值而非向量
    loss_ = loss.sum().item()

    # pred_ ==> [bz, seq_len]
    pred_ = pred.argmax(dim=-1)
    acc = pred_.eq(tgt_real)
    # acc计算比例需要算占总比多少
    # 画图时必须要item()
    acc = (acc.sum()/ tgt.size(0)/ tgt.size(1)).item()

    loss.backward()
    optimizer.step()

    return loss_, acc 
```

注意这个是随机的，所以每一次的结果都不一样

```python
# 检查train_step()的效果
criterion = nn.CrossEntropyLoss()
batch_src, batch_tgt = next(iter(train_dataloader))
train_step(batch_src, batch_tgt, criterion)

# (8.144109725952148, 0.0)
```

```python
def validation_step(inp, tgt, criterion):
    tgt_input = tgt[:,:-1]
    tgt_real = tgt[:,1:]

    inp = inp.to(device)
    tgt_input = tgt_input.to(device)
    tgt_real = tgt_real.to(device)

    model.eval()

    with torch.no_grad():
        pred, _, _, _ = model(inp, tgt_input)
        val_loss = criterion(pred.view(-1, pred.size(-1)), tgt_real.contiguous().view(-1))
        val_loss = val_loss.sum().item()

        pred_ = pred.argmax(dim=-1)
        val_acc = pred_.eq(tgt_real)
        val_acc = (val_acc.sum()/ tgt.size(0)/ tgt.size(1)).item()
    return val_loss, val_acc
```

***
### <div id='train'>模型训练</div>

```python
EPOCHS = 20

print_trainstep_every = 50

lr_scheduler = CustomSchedule(optimizer, warm_steps=4000)

# 存储数据
df_history = pd.DataFrame(columns=["epoch","loss","acc","val_loss","val_acc"])

# 打印时间
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m_%d %H:%M:%S')
    print('\n' + "=========="*4 + '%s'%nowtime + "=========="*4)
```

```python
save_dir = "train"
def train_model(epochs, train_dataloader, val_dataloader, print_every):
    starttime = time.time()
    print('\n' + "=========="*4 + "start training" + "=========="*4)
    best_acc = 0.81
    for epoch in range(1, epochs + 1):

        loss_sum = 0
        acc_sum = 0

        for step, (inp, tgt) in enumerate(train_dataloader, start=1):
            loss, acc = train_step(inp, tgt, criterion)
            loss_sum += loss
            acc_sum += acc 

            # 打印batch级别信息
            if step % print_every == 0:
                print('*' * 8, f'[step = {step}] loss: {loss_sum / step:.3f}, {"acc"}: {acc_sum / step:.3f}')
            
            # 更新学习率
            lr_scheduler.step()
        # 一个epoch结束，做一次验证
        val_loss_sum = 0
        val_acc_sum = 0
        for val_step, (inp, tgt) in enumerate(val_dataloader, start=1):
            val_loss, val_acc = validation_step(inp, tgt, criterion)
            val_loss_sum += val_loss
            val_acc_sum += val_acc 
        
        # 记录收集一个epoch的信息
        # 与列对应
        # epoch从0开始
        record = (epoch, loss_sum/step, acc_sum/step, val_loss_sum/val_step, val_acc_sum/val_step) 
        df_history.loc[epoch-1] = record
        # 打印epoch级别的日志
        print('EPOCH = {} loss: {:.3f}, {}: {:.3f}, val_loss: {:.3f}, val_{}: {:.3f}'.format(
        record[0], record[1], "acc", record[2], record[3], "acc", record[4]))
        printbar()
        current_acc_avg = val_acc_sum / val_step 
        # 若大于基础正确率，则保存
        if current_acc_avg > best_acc:
            best_acc = current_acc_avg
            checkpoint = save_dir + '{:03d}_{:.2f}_ckpt.tar'.format(epoch, current_acc_avg)
            
            # 若只是普通保存，则只会保留基础网络结构而非具体参数
            model_sd = copy.deepcopy(model.state_dict())
            torch.save({
                'loss': loss_sum / step,
                'epoch': epoch,
                'net': model_sd,
                'opt': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }, checkpoint)
        print('finishing training...')
    
    # 时间记录
    endtime = time.time()
    time_elapsed = endtime - starttime
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return df_history
```

```python
# 开始训练
df_history = train_model(EPOCHS, train_dataloader, val_dataloader, print_trainstep_every)
print(df_history)
```

截取部分，如下：

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/res.png)


接下来把训练结果可视化：

```python
def plot_metric(df_history, metric):
    plt.figure()

    train_metrics = df_history[metric]
    val_metrics = df_history["val_" + metric]

    # epochs变为列表，才能画
    epochs = range(1, len(train_metrics) + 1)

    plt.plot(epochs, train_metrics, "bo--")
    plt.plot(epochs, val_metrics, "ro--")

    plt.title("Training and validation " + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric,"val_" + metric])
    plt.savefig(metric + " conv" + ".png")
    plt.show()

plot_metric(df_history, "loss")
plot_metric(df_history, "acc")
```

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/1.png)

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/2.png)

***
### <div id='eval'>评估</div>

加载模型

```python
# 具体看保存哪个，需要自行修改
checkpoint = "/content/train015_0.82_ckpt.tar"
print("checkpoint:",checkpoint)

ckpt = torch.load(checkpoint)

transformer_sd = ckpt["net"]

reload_model = Transformer().to(device)
reload_model.load_state_dict(transformer_sd)
```

```python
def tokenizer_encode(tokenize, sentence, vocab):
    sentence = normalizeString(sentence)

    sentence = tokenize(sentence)
    sentence = ["<start>"] + sentence + ["<end>"]
    sentence_ids = [vocab.stoi[token] for token in sentence]
    return sentence_ids

def tokenizer_decode(sentence_ids, vocab):
    sentence = [vocab.itos[id] for id in sentence_ids if id < len(vocab)]
    return " ".join(sentence)
```

```python
# 只有一个句子，不需要加pad
s = 'je pars en vacances pour quelques jours .'
print(tokenizer_encode(tokenizer, s, SRC_TEXT.vocab))

s_ids = [3, 5, 251, 17, 365, 35, 492, 390, 4, 2]
print(tokenizer_decode(s_ids, SRC_TEXT.vocab))
print(tokenizer_decode(s_ids, TGT_TEXT.vocab))

# [3, 5, 251, 17, 365, 35, 492, 390, 4, 2]
# <start> je pars en vacances pour quelques jours . <end>
# <start> i tennis very forgetful me helping bed . <end>
```
评估函数

```python
def evaluate(inp_sentence):
    reload_model.eval()
    
    inp_sentence_ids = tokenizer_encode(tokenizer, inp_sentence, SRC_TEXT.vocab)
    # =>[b=1, inp_seq_len=10]
    encoder_input = torch.tensor(inp_sentence_ids).unsqueeze(dim=0)

    # 预估时一句对一句，decode_input为<start>
    decoder_input = [TGT_TEXT.vocab.stoi["<start>"]]
    # =>[b=1, inp_seq_len=1]    
    decoder_input = torch.tensor(decoder_input).unsqueeze(0)

    encoder_input = encoder_input.to(device)
    decoder_input = decoder_input.to(device)

    with torch.no_grad():
        for i in range(MAX_LENGTH + 2):
            # pred ==> [b=1, 1(len_q), tgt_vocab_size]
            pred, enc_self_attns, dec_self_attns, dec_enc_attns = reload_model(encoder_input, decoder_input)
            # 最后一个词
            # pdb.set_trace()
            pred = pred[:,-1:,:]
            # pred_ids ==> [1,1]
            pred_ids = torch.argmax(pred, dim=-1)

            if pred_ids.squeeze().item() == TGT_TEXT.vocab.stoi["<end>"]:
              return decoder_input.squeeze(dim=0), dec_enc_attns
            
            # [b=1, tgt_seq_len=1] ==> [b=1, tgt_seq_len=2]
            # deocder_input不断变长
            decoder_input = torch.cat([decoder_input, pred_ids],dim=-1)
    return decoder_input.squeeze(dim=0), dec_enc_attns
```

```python
s = 'je pars en vacances pour quelques jours .'
s_targ = 'i m taking a couple of days off .'
pred_result, attention_weights = evaluate(s)
pred_sentence = tokenizer_decode(pred_result, TGT_TEXT.vocab)
print('real target:', s_targ)
print('pred_sentence:', pred_sentence)

# real target: i m taking a couple of days off .
# pred_sentence: <start> i m taking a couple of days off .
```

```python
# 批量翻译
sentence_pairs = [
    ['je pars en vacances pour quelques jours .', 'i m taking a couple of days off .'],
    ['je ne me panique pas .', 'i m not panicking .'],
    ['je recherche un assistant .', 'i am looking for an assistant .'],
    ['je suis loin de chez moi .', 'i m a long way from home .'],
    ['vous etes en retard .', 'you re very late .'],
    ['j ai soif .', 'i am thirsty .'],
    ['je suis fou de vous .', 'i m crazy about you .'],
    ['vous etes vilain .', 'you are naughty .'],
    ['il est vieux et laid .', 'he s old and ugly .'],
    ['je suis terrifiee .', 'i m terrified .'],
]


def batch_translate(sentence_pairs):
    for pair in sentence_pairs:
        print('input:', pair[0])
        print('target:', pair[1])
        pred_result, _ = evaluate(pair[0])
        pred_sentence = tokenizer_decode(pred_result, TGT_TEXT.vocab)
        print('pred:', pred_sentence)
        print('')

batch_translate(sentence_pairs)
```
截取部分

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/2.png)

最终可视化注意力权重

```python
# 可视化attenton 这里我们只展示...block2的attention，即[b, num_heads, tgt_seq_len, inp_seq_len]
# attention: {'decoder_layer{i + 1}_block1': [b, num_heads, tgt_seq_len, tgt_seq_len],
#             'decoder_layer{i + 1}_block2': [b, num_heads, tgt_seq_len, inp_seq_len], ...}
# sentence: [seq_len]，例如：'je recherche un assistant .'
# pred_result: [seq_len]，例如：'<start> i m looking for an assistant .'
# layer: 表示模型decoder的N层decoder-layer的第几层的attention，形如'decoder_layer{i}_block1'或'decoder_layer{i}_block2'
def plot_attention_weights(attention, sentence, pred_sentence, layer):
    sentence = sentence.split()
    pred_sentence = pred_sentence.split()
    # pdb.set_trace()
    fig = plt.figure(figsize=(16, 8))

    # block2 attention[layer] => [b=1, num_heads, targ_seq_len, inp_seq_len]
    # attention为列表，长度为层数6
    attention = torch.squeeze(attention[layer], dim=0) # => [num_heads, targ_seq_len, inp_seq_len]

    for head in range(attention.shape[0]):
        # 111是单个整数编码的子绘图网格参数。例如，“111”表示“1×1网格，第一子图”，“234”表示“2×3网格，第四子图”

        ax = fig.add_subplot(2, 4, head + 1)  
        cax = ax.matshow(attention[head].cpu(), cmap='viridis')  # 绘制网格热图，注意力权重
        # fig.colorbar(cax)#给子图添加colorbar（颜色条或渐变色条）

        fontdict = {'fontsize': 10}

        # 设置轴刻度线
        ax.set_xticks(range(len(sentence)+2))  # 算上start和end
        ax.set_yticks(range(len(pred_sentence)))

        ax.set_ylim(len(pred_sentence) - 1.5, -0.5)  # 设定y座标轴的范围

        # 设置轴
        ax.set_xticklabels(['<start>']+sentence+['<end>'], fontdict=fontdict, rotation=90)  # 顺时间旋转90度
        ax.set_yticklabels(pred_sentence, fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head + 1))
    plt.tight_layout()
    plt.show()


def translate(sentence_pair, plot=None):
    print('input:', sentence_pair[0])
    print('target:', sentence_pair[1])
    pred_result, attention_weights = evaluate(sentence_pair[0])
    # print('attention_weights:', attention_weights[0])
    # pdb.set_trace()
    pred_sentence = tokenizer_decode(pred_result, TGT_TEXT.vocab)
    print('pred:', pred_sentence)
    print('')

    if plot:
        plot_attention_weights(attention_weights, sentence_pair[0], pred_sentence, plot)


translate(sentence_pairs[0], plot=1)
translate(sentence_pairs[2], plot=2)
```

```python
# input: je pars en vacances pour quelques jours .
# target: i m taking a couple of days off .
# pred: <start> i m taking a couple of days off .
```
![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/attn.png)

```python
# input: je recherche un assistant .
# target: i am looking for an assistant .
# pred: <start> i am looking for an assistant .
```

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/attn_.png)

***
### <div id='discuss'>讨论</div>

接下来列举了一些有趣的讨论：

- 卷积代替全连接？

这里使用的是1 x 1的卷积，不会改变shape，只会改变通道数，具体关于这方面的阐述，请移步[CNN](https://github.com/sherlcok314159/ML/blob/main/NN/CNN/cnn.md)，这样的话就是不同通道之间进行线性组合，如原来的通道数由2变为1，则原来两个相加，卷积的时候也是进行矩阵乘法运算。全连接层也是乘上参数矩阵，这两者应该差不多，详见下方：

```python
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)
```

以下是我训练的结果，其实结果证明确实是差不多的，前两张为全连接层

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/1.png)

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/2.png)

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/conv1.png)

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/conv2.png)

- Bert作为Encoder?

写完transformer的教程本打算直接硬上bert来train一发，看了论文才发现事情没那么简单。ICLR2020这篇Incorporating BERT into Neural Machine Translation（参考文献第二个）详细讲了这件事，总结来说就是一般情况使用bert结果不会好反倒糟，不值得花气力，当然，如果要在词料丰富和词料贫乏的语言之间构造翻译器，那么bert作为encoder可能有奇效。

- epoch越多越好？

我设了两组对照，分别为20和40，两组的原因是40中包括了其他大于20的，其实到20差不多已经饱和了，为了节省时间，还是设置20来的划算


![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/20.png)

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/40.png)

***
### <div id='references'>参考文献</div>

https://arxiv.org/pdf/2002.06823.pdf

[Step By Step](https://blog.csdn.net/xixiaoyaoww/article/details/105683495?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522162246804316780271557962%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=162246804316780271557962&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_v2~rank_v29-1-105683495.nonecase&utm_term=%E7%BF%BB%E8%AF%91&spm=1018.2226.3001.4450)

https://blog.csdn.net/u010366748/article/details/111269231
