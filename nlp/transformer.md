
### Transformer

参考论文(https://arxiv.org/abs/1706.03762)

**章节**
- [Reasons](#reasons)
- [Self-Attention](#self_attention)
    - [Multi-Headed](#multi)
- [Positional Encoding](#positional)
- [Add & Normalize](#add)
- [Source Code Explanation](#code)

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

首先将词编码成向量，总不大可能直接将词输进去。其次定义三个矩阵，W^Q,W^K,W^V这三个矩阵分别代表了Query,Key,Value，分别代表去查询的，被查询的以及实际的特征信息。W的意思是权重的意思，可以类比于梯度下降，权重一般都是初始化一个参数，然后通过训练得出。最后用词向量与矩阵分别做**点积**即可得到q1,q2,k1,k2,v1,v2。

用q1,q2分别与k1,k2的转置做**点积**，q代表的要查的，而k是被查的。如此便可得到两者的关系了。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/qkv_2.png)

这里简要解释为什么用内积，在坐标系中，当两个向量垂直时，其关系最小，为0。其实就是cos为0。假如a1,a2两个向量很接近，那么，它们之间的夹角会很小，可知cos就很大，代表联系就越紧密。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/values.png)

外面用softmax起着归一化的作用将具体的数值转化为，概率，更加直观。

当然，两者做点积之后还需要除以矩阵K的维度开根号，Q，K，V矩阵维度是q x d_k，p x d_k，p x d_v，softmax是沿着p维进行的，但是很多时候大方差会导致数值分布不均匀，经过softmax之后就会大的愈大，小的愈小，这里除以一个矩阵K的维度其实类似于一个归一化，让它的**方差趋向于1，分布均匀一点**，所以在原paper里面又叫做**Scaled Dot-Product Attention**。

例如 v = 0.36v1 + 0.64v2，v1,v2是矩阵V里的。

既然都是矩阵运算，那么都是可以**并行加速**的。
***
**<div id='multi'>Multi-headed Attention</div>**

理解了自注意力，那么什么是多头注意力呢？

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/cnn.png)
------------------------[图片来源](https://www.researchgate.net/publication/325924260_A_Simple_Fusion_Of_Deep_And_Shallow_Learning_For_Acoustic_Scene_Classification/figures?lo=1)----------------------------------------------------------------------------

类比一下[CNN](../NN/CNN/cnn.md)，当我们不断提取特征的时候会用到一个叫卷积核的东西（filter），我们为了最大化提取有效特征，通常选择**一组卷积核**来提取，因为不同的卷积核对图片的不同区域的注意力不同。
比如，我们要识别一张鸟的图片，可能第一个卷积核更关注鸟嘴，第二个卷积核更关注鸟翅膀等等。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/qkv_4.png)

在Transformer中也是如此，不同的Q，K，V矩阵得出的特征关系也不相同，同样，不一定一组Q，K，V提取到的关系能解决问题，所以保险起见，我们用多组。这里可以把一组Q，K，V矩阵类比为一个卷积核，最后再通过全连接层进行拼接降维。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/qkv_3.png)

****
**<div id='positional'>Positional Encoding</div>**

为什么要进行位置编码呢？

我上面阐述的注意力机制是不是只说了某个词与它所处的句子之间的关系，但是在实际自然语言处理中，只知道这个词与句子的关系而不知道它在哪个位置是不行的。

里面用了正余弦函数，具体做什么我们在源码节细讲。

***

**<div id='add'>Add & Normalize</div>**

