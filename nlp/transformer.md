
### Transformer

参考论文(https://arxiv.org/abs/1810.04805)

**章节**
- [Reasons](#reasons)
- [Self-Attention](#self_attention)
    - [Multi-Headed](#multi_headed)
- [Add & Normalize](#add)
- [Positional Encoding](#positional)

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/transformer.png)


**<div id='reasons'>Reasons For Transformer</div>**

新模型的产生往往是为了旧模型的问题所设计的。那么，原始模型的问题有哪些呢？

**1、无法并行运算**

在transformer之前，大部分应该都是RNN，下面简单介绍一下RNN


可以看出，在计算X2加进去吐出来的值的时候必须要先把X1得出的参数值与X2放在一起才行。换句话说，RNN的计算是必须**一个接一个**，并不存在并行运算。如果不能并行运算，那么时间和计算成本均会增加。

**2、语义表述不清**

