
### Transformer

参考论文(https://arxiv.org/abs/1706.03762)

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

在transformer之前，大部分应该都是[RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network)，下面简单介绍一下RNN

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/RNN.png)


可以看出，在计算X2加进去吐出来的值的时候必须要先把X1得出的参数值与X2放在一起才行。换句话说，RNN的计算是必须**一个接一个**，并不存在并行运算。如果不能并行运算，那么时间和计算成本均会增加。

**2、语义表述不清**

传统的[word2vec](https://en.wikipedia.org/wiki/Word2vec)通过将词语转换为坐标形式，并且根据**距离之远近决定两个词意思的相近程度。** 但是在NLP任务中，尽管是同一个词语，但在不同语境下代表的意思很有可能不同。例如，你今天方便的时候，来我家吃饭吧和我肚子不舒服，去厕所方便一下这两句中方便的意思肯定不一样。可是，word2vec处理之后坐标形式就固定了。

**3、突出对待**

在一个句子中，当我们需要强调某一个部分的时候，word2vec无法为我们做到这些。比如，

The cat doesn't eat the cake because it is not hungry.

The cat doesn't eat the cake because it smells bad.

第一个句子中it强调的是the cat，在第二个句子中it强调的是the cake。

***
**<div id='self_attention'>Self-Attention</div>**

attention的意思是我们给有意义的内容配以较高的权重，那么自己与自己做attention是什么意思？

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/self_attention.png)

比如这个句中的"green"，self的意思就是说拿这个词与整个句子其他的词语分别算相似程度。如此便考虑了词与上下文和句子整体之间的关系。当然，自己与自己肯定是联系最紧密。我们的目的是让机器能够判定出某个词语与整个句子之间的关系。

那么，如何计算self-attenion呢？

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/qkv.png)

------------------------------------------------------------------------[图片来源](https://www.bilibili.com/video/BV1NJ411o7u3?p=5)----------------------------------------------------------------------------

首先将词编码成向量，总不大可能直接将词输进去。其次定义三个矩阵，$W^Q$,$W^K$,$W^V$这三个矩阵分别代表了Query,Key,Value，分别代表去查询的，被查询的以及实际的特征信息。