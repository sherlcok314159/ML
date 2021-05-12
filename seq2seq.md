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
        self.gru = nn.GRU(hidden_size,hidden_size)

    def forward(self,input,hidden):
        embedded = self.embedding(input).view(1,1,-1)
        output = embedded
        output,hidden = self.gru(output,hidden)
        return output,hidden
    
    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size,device=device)
```

