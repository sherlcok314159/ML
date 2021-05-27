### Transformer之翻译篇

章节

- [MASK机制](#mask)
- [Encoder](#encoder)
- [Decoder](#decoder)
- [训练模型](#train)


### <div id='mask'>MASK机制</div>

原始的句子首先需要转换为词表中的索引，然后进入词嵌入层。举个例子，假如某个时间步长上输入句子为`"I love u"`，src_vocab（源语言词表）为`{"BOS":0,"EOS":1,"I":2,"love":3,"u":4}`，那么输入句子变为`[[2, 3, 4, 1]]`，接下来进入词嵌入层，目前我们的词表只有5个词，所以`embed = nn.Embedding(5, 6)`，用一个6维向量表示每一个词，如下所示：

```python
import torch
import torch.nn as nn
t = torch.tensor([[2, 3, 4, 1]])
embed = nn.Embedding(5, 6)
print(embed(t))
tensor([[[-0.5557,  0.9911, -0.2482,  1.5019,  0.9141,  0.0697],
         [-1.5058, -0.4237,  1.1189, -0.7472, -0.9834, -1.2829],
         [-0.8558,  0.4753,  0.0555, -0.3921, -0.0232,  0.2518],
         [-0.3563,  0.7707, -0.8797,  0.6719, -0.4903,  0.0508]]],
       grad_fn=<EmbeddingBackward>)
```

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

做自注意力的时候，q和k都是q，接下来实例看一下，`True`的地方代表是填充的。

```python
q = torch.tensor([[2, 3, 4, 1, 0, 0]])
print(get_attn_pad_mask(q, q))
# tensor([[[False, False, False, False,  True,  True],
#         [False, False, False, False,  True,  True],
#         [False, False, False, False,  True,  True],
#         [False, False, False, False,  True,  True],
#         [False, False, False, False,  True,  True],
#         [False, False, False, False,  True,  True]]])
```
