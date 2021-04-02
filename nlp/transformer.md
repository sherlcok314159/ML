
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


