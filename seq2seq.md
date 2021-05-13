### 基于Seq2Seq模型实现法语向英语的翻译

参考文章(https://pytorch123.com/FifthSection/Translation_S2S_Network/)

**章节**

- [简介](#abstract)
- [Seq2Seq模型](#seq2seq)
- [文本预处理](#preprocess)


***
**<div id='abstract'>简介</div>**

本文基于PyTorch实现seq2seq模型来实现法语向英语的翻译，会带你了解从seq2seq模型的主要原理，在实战中需要对文本的预处理，到训练和可视化一个完整的过程。读完本文，不仅可以通过这个项目来了解PyTorch的语法以及RNN（GRU）的操作，而且可以真正明白seq2seq模型的内涵。

***

**<div id='seq2seq'>Seq2Seq模型</div>**

seq2seq模型顾名思义就是序列到序列模型（sequence to sequence）。我们可以把这个模型当做一顶魔法帽，帽子的输入和输出均为序列，举个例子，翻译就是一个序列到序列的任务，输入法文句子，输出则是英文句子（见下图）。

![](https://github.com/sherlcok314159/ML/blob/main/NN/Images/seq2seq.png)

seq2seq模型主要由Encoder以及Decoder这两大部分组成，这两者分别执行将序列编码以及解码的工作，而在这个项目里面我们以GRU作为主要组成部分。GRU可以近似为LSTM的变种，比LSTM结构更简单，计算更加方便，而实际效果和LSTM相差无几。另外，关于LSTM 以及 GRU的详细解释可以看我的另一篇[RNN综述](NN/RNN/rnn.md)，本文偏重代码实现而非原理解释。

首先来看一下Encoder，首先会对输入进行一个词嵌入，词嵌入的好处是可以对长短不一的输入进行统一长度，方便计算。GRU每一个时间步上的输入有两个，一个是上一个时间步的隐藏状态（hidden_state），另一个是当前时间步的文本输入。

![](https://github.com/sherlcok314159/ML/blob/main/NN/Images/encoder.png)

seq2seq模型中一个一个的单词其实是每一个时间步的输入和输出，而在训练模型时，单词是转换为索引（注意这个索引其实是在预处理部分决定的，常见的索引有单词在整个单词集中出现的顺序），在torch中还要转成tensor格式，比如说第一个单词，它的索引是2，那么它其实是tensor([2])。Embedding层正是我们对于输入的每一个词进行词嵌入处理，虽然我们每一次只输入一个单词，但我们在初始化的时候会将训练集中所有的单词一起预先处理好，然后当每一个单词索引进来之后，就好比查字典一样，找到自己对应词嵌入之后的tensor。注意，这里不是把所有的单词直接送进去，每一个单词索引有一个Embedding，这里的意思是告诉Embedding层一共有多少个embedding。这样的好处是直接固定了，不会随着不断输入而改变Embedding层参数。

所以，Embedding层第一个参数其实是训练集中单词的数量，第二个参数指的是每一个单词拥有多少维的编码。单词索引送进去了，tensor([2])，假设Embedding层参数是(2,4)，则经过词嵌入后的结果是tensor([[0.1,2.1,3.1,0.9]])，会发现第二个size其实是需要嵌入的维度。

包的引入

```python
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
```

```python
# 若无GPU，则CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        # 调用父类初始化方法
        super(EncoderRNN, self).__init__()
        # 初始化必须的变量
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)        # gru的输入为三维，两个参数均指的是最后一维的大小
        # tensor([1,1,hidden_size])
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        # embedded.size() ==> tensor([1,1,hidden_size])
        # -1的好处是机器会自动计算
        # 这里用view扩维的原因是gru必须接受三维的输入
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded        
        output, hidden = self.gru(output, hidden)
        return output, hidden
    
    def initHidden(self):
        # 初始化隐层状态全为0
        # hidden ==> tensor([1,1,hidden_size])
        return torch.zeros(1, 1, self.hidden_size, device=device)

```

![](https://github.com/sherlcok314159/ML/blob/main/NN/Images/decoder.png)

接下来介绍Decoder，在本文中仅使用Encoder中最后一个输出的hidden来作为Decoder的初始的hidden，因为编码器最后一个hidden常常含有整个序列的上下文信息，有时会被称为上下文变量。

这里的第一个文本输入其实是\<sos>（start of sentence），与Encoder不同的是，这里经过词嵌入之后还做了relu处理，增强模型非线性的表达能力。

输入会经过一个softmax来获得一个概率分布，最后取最大概率的那个作为当前预测的结果

上一个时间步的hidden总会作为当前时间步的hidden输入，而当前时间步的文本输入是上一个时间步的预测结果。



```python
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        # input_features ==> hidden_size
        # output_features ==> output_size
        self.out = nn.Linear(hidden_size, output_size)        # Log(Softmax(X))
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        # output.size() ==> [1,1,hidden_size]
        # output的第一个1是我们用以适合gru输入扩充的
        # 所以用output[0]选取前面的
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
```


在刚刚的基础上进行升级，在Decoder上引入注意力机制，对比一下，没加注意力之前，Decoder是直接接受全部的Encoder输出，而attention加入之后可以更准确地聚焦到Encoder输出的不同部分，具体是用注意力权重矩阵去乘以Encoder的输出向量用以创建加权组合，从而帮助Decoder选择正确的输出。

实现时将Decoder的文本输入和隐藏状态作为输入，分别对应图中的input,prev_hidden（上一个时间步的隐藏状态）。文本输入进来经过词嵌入之后应用了dropout，可以一定程度减少模型过拟合，增强模型的泛化能力。通过前馈层attn之后进行softmax处理再和Encoder的输出矩阵做点乘处理，再拼接起来加一个relu。注意，上一个时间步的隐藏状态会继续作为gru的状态输入。

![](https://github.com/sherlcok314159/ML/blob/main/NN/Images/attDecoder.png)

```python
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # 因为会将prev_hidden和embedded在最后一个维度
        # 即hidden_size，进行拼接，所以要*2
        # max_length用以统一不同长度的句子分配的注意力
        # 最大长度句子使用所有注意力权重，较短只用前几个
        self.attn = nn.Linear(self.hidden_size*2,self.max_length)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # 因为第一维只是适应模型输入扩充的
        # 所以拼接时，只需要取后面两个维度
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)        # bmm ==> batch matrix multiplication
        # e.g. a.size() ==> tensor([1,2,3])
        # b.size() ==> tensor([1,3,4])
        # torch.bmm(a,b).size() ==> tensor([1,2,4])  
        # 第一维度不变，其他两维就当作矩阵做乘法
        # unsqueeze(0)用以在在第一维扩充维度
        # attn_applied赋予encoder_outputs不同部分不同权重
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

```
***

了解完模型的主要架构，接下来了解一下模型输入数据的处理

**<div id='preprocess'>文本预处理</div>**

> 你训练的模型不过是无限逼近你data所能提供的上界而已

在NLP中，对数据的前期处理也是十分重要的，大的思路就是统一长度，变为数值。

以下两个分别代表一个序列的开始和结束

```python
SOS_token = 0
EOS_token = 1
```

对语言进行初步处理并返回Lang对象

```python
class Lang:
    def __init__(self,name):
        self.name = name
        # 形如 {"hello" : 3}
        self.word2index = {}
        # 统计每一个单词出现的次数
        self.word2count = {}
        self.index2word = {0:"SOS",1:"EOS"}
        # 统计训练集出现的单词数
        self.n_words = 2 # SOS 和 EOS已经存在了

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
将unicode字符转换为纯Ascii

```python
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
```

```python
def normalizeString(s):
    # 转码之后变小写切除两边空白
    s = unicodeToAscill(s.lower().strip())
    # 匹配.!?，并在前面加空格
    s = re.sub(r"([.!?])",r" \1",s)
    # 将非字母和.!?的全部变为空白
    s = re.sub(r"[^a-zA-Z.!?]+",r" ",s)
    return s
```
```python
def readLangs(lang1,lang2,reverse=False):
    print("Reading lines...")

    # 读取文件并分为几行
    # 每一对句子最后会有个换行符\n
    # lines ==> ['Go.\tVa !', 'Run!\tCours\u202f!'...]
    lines = open("填你的数据路径",encoding = "utf-8").read().strip().split("\n")

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

对上面处理完的pair对进行两个判断，是否每一个pair长度都小于MAX_LENGTH，第二个pair是否以eng_prefixes开头（本文会进行反转），这样可以减少训练量，加快收敛。

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
```

接着进行整合

```python
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
```

接下来创建对句子里的单词的索引，并转为tensor形式

```python
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)
```

***