### Transformer之翻译篇

章节

- [MASK机制](#mask)
- [Encoder](#encoder)
- [Decoder](#decoder)
- [训练模型](#train)


### <div id='mask'>MASK机制</div>

原始的句子首先需要转换为词表中的索引，然后进入词嵌入层。举个例子，假如某个时间步长上输入句子为`"I love u"`，src_vocab（源语言词表）为`{"BOS":0,"EOS":1,"I":2,"love":3,"u":4}`，`BOS`和`EOS`代表句子的开头和末尾，那么输入句子变为`[[2, 3, 4, 1]]`，接下来进入词嵌入层，目前我们的词表只有5个词，所以`embed = nn.Embedding(5, 6)`，用一个6维向量表示每一个词，如下所示：

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

但现实中往往还要复杂的多，其中最重要的问题即为不定长，在RNN中序列可以不定长，可是在transformer中是需要定长的，这意味着需要设置一个最大长度，小于最大长度的全部补0，这些补0的就被称为`<PAD>`，对应的索引常为0，src_vocab扩充为`{"PAD":0,"BOS":1,"EOS":2,"I":3,"love":4,"u":5}`，假如最大长度为6，那么输入句子对应的索引变为`[[2, 3, 4, 1, 0, 0]]`。长度统一之后，新的问题产生了：那些补0的地方机器并不知道，在做注意力运算的时候还是要加入运算，这样结果带有一定的误导性，为了减免填充的地方对注意力运算带来影响，在实际运用中用`MASK`遮住补0的地方。

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

不妨继续上面的例子，q依旧是`[[2, 3, 4, 1, 0, 0]](tensor形式)`，此时q变成了解码器里面的输入，不仅要遮挡住PAD，同时还得让每个位置只能瞻前不能顾后

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

### <div id='encode'>Encoder</div>

