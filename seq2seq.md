### 基于Seq2Seq模型实现法语向英语的翻译

参考文章(https://pytorch123.com/FifthSection/Translation_S2S_Network/)

**章节**

- [简介](#abstract)
- [Seq2Seq模型](#seq2seq)



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


```python
# 若无GPU，则CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def EncoderRNN(nn.Module):
    def __init__(self,input_size,hidden_size):
        # 调用父类初始化方法
        super(EncoderRNN,self).__init__()
        # 初始化必须的变量
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size,hidden_size)
        # gru的输入为三维，两个参数均指的是最后一维的大小
        # tensor([1,1,hidden_size])
        self.gru = nn.GRU(hidden_size,hidden_size)

    def forward(self,input,hidden):
        # embedded.size() ==> tensor([1,1,hidden_size])
        # -1的好处是机器会自动计算
        # 这里用view扩维的原因是gru必须接受三维的输入
        embedded = self.embedding(input).view(1,1,-1)
        output = embedded
        output,hidden = self.gru(output,hidden)
        return output,hidden
    
    def initHidden(self):
        # 初始化隐层状态全为0
        # hidden ==> tensor([1,1,hidden_size])
        return torch.zeros(1,1,self.hidden_size,device=device)
```

![](https://github.com/sherlcok314159/ML/blob/main/NN/Images/decoder.png)

接下来介绍Decoder，在本文中仅使用Encoder中最后一个输出的hidden来作为Decoder的初始的hidden，因为编码器最后一个hidden常常含有整个序列的上下文信息，有时会被称为上下文变量。

这里的第一个文本输入其实是"<bos>"（beginning of setence），与Encoder不同的是，这里经过词嵌入之后还做了relu处理，增强模型非线性的表达能力。

输入会经过一个softmax来获得一个概率分布，最后取最大概率的那个作为当前预测的结果

上一个时间步的hidden总会作为当前时间步的hidden输入，而当前时间步的文本输入是上一个时间步的预测结果。



```python
class DecoderRNN(nn.Module):
    def __init__(self,hidden_size,output_size):
        super(DecoderRNN,self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # Log(Softmax(X))
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self,input,hidden):
        output = self.embedding(input).view(1,1,-1)
        output = F.relu(output)
        output,hidden = self.gru(output,hidden)
        output = self.softmax(self.out(output[0]))
        return output,hidden

    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size,device=device)
```