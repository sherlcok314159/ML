### Transformer之翻译篇

章节

- [数据预处理](#preproces)
- [MASK机制](#mask)
- [Encoder](#encoder)
- [模型搭建](#model)
- [训练函数](#train)

### <div id='preprocess'>数据预处理</div>

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

***

### <div id='mask'>MASK机制</div>

源码在[colab](https://colab.research.google.com/drive/1CILp7vwm8bZy6dOnRuwPeujP3-Mdm67z?usp=sharing)上，数据集若要自己下载[data](../RNN/eng-fra.txt)

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

```python
Tensor = torch.Tensor
def positional_encoding(X, num_features, dropout_p=0.1, max_len=512) -> Tensor:
    r'''
        给输入加入位置编码
    参数：
        - num_features: 输入进来的维度
        - dropout_p: dropout的概率，当其为非零时执行dropout
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
    return dropout(X)
```
- MASK 

```python
# 返回True和False组成的MASK
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

# 返回0和1组成的MASK
def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    # 转换为tensor，并且修改数据类型：dtype=torch.uint8
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask
```

- ScaledDotProductAttention

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    
    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2))
        # 1的位置全部填上"-inf"
        scores.masked_fill(attn_mask, float("-inf"))
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn
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
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]
```


- PoswiseFeedForwardNet

与以往不同的是，这里用1 x 1的卷积来代替全连接层

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

- EncoderLayer

在Encode的时候我们只需要使用pad_mask即可，PyTorch的transformer库有很多mask，如src_mask，src_key_padding_mask，前者按照官方的意思是让某些位置被忽略，后者是与pad_mask等价的，在训练Encoder的时候，两者等价，只需要一个src_key_padding_mask即可，这里即为`enc_self_attn_mask`

```python
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        # 自注意因而q,k,v均为enc_inputs
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) 
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
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

    def forward(self, enc_inputs): # enc_inputs : [batch_size x source_len]
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = positional_encoding(enc_outputs,d_model)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
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

    def forward(self, dec_inputs, enc_inputs, enc_outputs): # dec_inputs : [batch_size x target_len]
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = positional_encoding(dec_outputs,d_model)        
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        # torch.gt严格大于
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
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
        dec_logits = self.projection(dec_outputs) # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
```

关于数据处理，因为数据集与RNN的seq2seq翻译模型一致，所以处理方式几乎相同，除了多了个`<PAD>`，这里不再赘述，同样这里只是Demo级，选取了特定前缀以及特定最大长度的。

```python
SOS_token = 1
EOS_token = 2


class Lang:
    def __init__(self,name):
        self.name = name
        # 形如 {"hello" : 3}
        self.word2index = {"PAD":0,"SOS":1,"EOS":2}
        # 统计每一个单词出现的次数
        self.word2count = {}
        # self.index2word = {0:"SOS",1:"EOS"}
        self.index2word = {0:"PAD",1:"SOS",2:"EOS"}
        # 统计训练集出现的单词数
        # self.n_words = 2 # SOS 和 EOS已经存在了
        self.n_words = 3

    def addSentence(self,sentence):
        # 第一行为 Go.  Va !
        # 前面是英语，后面是法语，中间用tab分隔
        for word in sentence.split(" "):
            self.addWord(word)
    
    def addWord(self,word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            # 用现有的总词数作为新的单词的索引
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
```
```python
# 将Unicode字符串转换为纯ASCII, 感谢https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# 小写，修剪和删除非字母字符


def normalizeString(s):
    # 转码之后变小写切除两边空白
    s = unicodeToAscii(s.lower().strip())
    # 匹配.!?，并在前面加空格
    s = re.sub(r"([.!?])",r" \1",s)
    # 将非字母和.!?的全部变为空白
    s = re.sub(r"[^a-zA-Z.!?]+",r" ",s)
    return s

def readLangs(lang1,lang2,reverse=False):
    print("Reading lines...")

    # 读取文件并分为几行
    # 每一对句子最后会有个换行符\n
    # lines ==> ['Go.\tVa !', 'Run!\tCours\u202f!'...]
    lines = open("填自己的文件路径",encoding = "utf-8").read().strip().split("\n")

    # 将每一行拆分成对并进行标准化
    # pairs ==> [["go .","va !"],...]
    pairs = [[normalizeString(s) for s in l.split("\t")] for l in lines]

    # 反向对，实例Lang
    # 源文件是先英语后法语
    # 换完之后就是先法后英
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    
    return input_lang,output_lang,pairs
```
```python
MAX_LENGTH = 10
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


# 留下符合条件的
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
# 随机输出pair对
print(random.choice(pairs))
print(input_lang.n_words)
print(output_lang.n_words)
```
```python
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def pad(X):
    if len(X) < MAX_LENGTH:
        X = X + [0] * (MAX_LENGTH - len(X))
    return X

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    indexes = pad(indexes)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0]).transpose(0,1)
    target_tensor = tensorFromSentence(output_lang, pair[1]).transpose(0,1)
    return (input_tensor, target_tensor)
```

***
### <div id='train'>训练函数</div>

train函数指的是一个句子对的训练函数，这里将loss变为20个epoch相加而来然后总的传播一次参数，相对来说更省时

```python
def train(enc_inputs,dec_inputs,tgt_inputs ,criterion):
  loss = 0
  for epoch in range(20):
      optimizer.zero_grad()
      outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
      loss += criterion(outputs,tgt_inputs.contiguous().view(-1))
  loss.backward()
  optimizer.step()
  return loss.item() / tgt_inputs.shape[1]
```
注意Decoder的输入是`<SOS>`，每次单独句子对的训练都会先将dec_inputs置为`<SOS>`

```python
if __name__ == "__main__":
  src_vocab = input_lang.word2index
  src_vocab_size = input_lang.n_words

  tgt_vocab = output_lang.word2index
  tgt_vocab_size = output_lang.n_words
   
  d_model = 512  # Embedding Size
  d_ff = 2048  # FeedForward dimension
  d_k = d_v = 64  # dimension of K(=Q), V
  n_layers = 6  # number of Encoder of Decoder Layer
  n_heads = 8  # number of heads in Multi-Head Attention
  
  model = Transformer()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  n_iters = 75000

  dec_inputs = torch.tensor([pad([0])])
  training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]
  for iter in range(1,n_iters + 1):

      training_pair = training_pairs[iter-1]
      input_tensor = training_pair[0]
      target_tensor = training_pair[1]
      loss = train(input_tensor, dec_inputs, target_tensor, criterion)

      if iter % 10 == 0:
          print(loss)
# 保留网络参数
torch.save(model.state_dict(),"transformer_parameters") 
```