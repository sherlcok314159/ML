
### Transformer

**章节**
- [Reasons](#reasons)
- [Self-Attention](#self_attention)
    - [Multi-Head Attention](#multi)
- [Positional Encoding](#positional)
- [Add & Norm](#add)
- [Feed Forward](#feed)
- [Residual Dropout](#drop)
- [Encoder To Decoder](#etd)
- [Shared Weights](#share)
- [Effect](#effect)
- [References](#references)

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


**4、长距离信息缺失**

尽管RNN处理序列问题很拿手，但如果一个句子很长，那么我要一直把句首的信息携带到句尾，距离越长，信息缺失的可能性就会变大。

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

注意力一共分为[additive attention](https://paperswithcode.com/method/additive-attention)和这里提到的dot product attention，那么为什么偏要用后者而非前者呢？

因为内积在实际中可以用高度优化矩阵运算，所以更快，同时空间利用率也很好。在此，简要解释一下内积可以表示两向量关系的原因，在坐标系中，当两个向量垂直时，其关系最小，为0。其实就是cos为0。假如a1,a2两个向量很接近，那么，它们之间的夹角会很小，可知cos就很大，代表联系就越紧密。

在K的维度很小的时候，两种注意力机制几乎一致，但是当K的维度上升之后，发现内积效果开始变差。其实，可以联系一下Softmax图像，当值变得很大的时候，梯度变化量几乎很难观察到，又被称为**梯度消失**问题，这是为什么做scale的第一个原因。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/values.png)

在论文中提到

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/paper_1.png)

两者做点积之后还需要除以矩阵K的维度开根号，Q，K，V矩阵维度是q x d_k，p x d_k，p x d_v，softmax是沿着p维进行的，但是很多时候大方差会导致数值分布不均匀，经过softmax之后就会大的愈大，小的愈小，这里除以一个矩阵K的维度其实类似于一个归一化，让它的**方差趋向于1，分布均匀一点**，这是第二个原因，所以在原paper里面又叫做**Scaled Dot-Product Attention**。

那为什么除以一个矩阵K的维度开根号能使方差变为1呢？首先对于随机分布的q，k，方差均为1，期望均为0。我们把特定的一个q_i,k_i看成X，Y。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/scaled.png)


那么q与k做点积之后的结果均值为0，方差为d_k。方差太大不稳定，所以除以矩阵K的维度开根号，参照链接(https://www.zhihu.com/question/339723385) 。


例如 v = 0.36v1 + 0.64v2，v1,v2是矩阵V里的。

既然都是矩阵运算，那么都是可以**并行加速**的。

self-attention除了可以捕获到句子语法特征外，还可以在长序列中捕获各个部分的**依赖关系**，而同样的处理用RNN和LSTM需要进行按照次序运算，迭代几次之后才有可能得到信息，而且距离越远，可能捕获到的可能性就越小。而self-attention极大程度上缩小了距离，更有利于利用特征。

***
**<div id='multi'>Multi-head Attention</div>**

理解了自注意力，那么什么是多头注意力呢？

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/cnn.png)

------------------------------------------------------------------------[图片来源](https://www.researchgate.net/publication/325924260_A_Simple_Fusion_Of_Deep_And_Shallow_Learning_For_Acoustic_Scene_Classification/figures?lo=1)----------------------------------------------------------------------------

类比一下[CNN](../NN/CNN/cnn.md)，当我们不断提取特征的时候会用到一个叫卷积核的东西（filter），我们为了最大化提取有效特征，通常选择**一组卷积核**来提取，因为不同的卷积核对图片的不同区域的注意力不同。
比如，我们要识别一张鸟的图片，可能第一个卷积核更关注鸟嘴，第二个卷积核更关注鸟翅膀等等。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/qkv_4.png)

在Transformer中也是如此，不同的Q，K，V矩阵得出的特征关系也不相同，同样，不一定一组Q，K，V提取到的关系能解决问题，所以保险起见，我们用多组。这里可以把一组Q，K，V矩阵类比为一个卷积核，最后再通过全连接层进行拼接降维。红线和绿线分别代表了两个不同的头在做自注意力时的情况，这里谷歌特地选了大不相同的两个头，其实在实践中，不同头的大部分关照的对象还是一样的，只有小部分不同，这也为后面为多头注意力减头提供了灵感。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/qkv_3.png)


****
**<div id='positional'>Positional Encoding</div>**

为什么要进行位置编码呢？

我上面阐述的注意力机制是不是只说了某个词与它所处的句子之间的关系，但是在实际自然语言处理中，只知道这个词与句子的关系而不知道它在哪个位置是不行的。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/position.png)

为什么用正余弦呢？因为这样的话，![](http://latex.codecogs.com/svg.latex?sin(x+k),cos(x+k))都可以被两者表示出来

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/_.png)

那么，既然有了公式，那位置编码是算出来的还是学出来的呢？其实算出来和学出来的效果差不多，但是考虑到算出来的可以接受更长的序列长度而不必受训练的干扰，所以在这个模型中，位置编码是通过公式算出来的。

这里用Toy Example来揭示位置编码的美妙之处：

假如我们现在有一个序列："I love nlp."，词典里有且仅有这三个词，且均按照顺序排列。一开始是找词典索引，那么结果变为：[0, 1, 2]（这里不算上batch），这个进入模型得经过一个词嵌入层，不妨假设每一个词用3维向量表示（公式里对应于![](http://latex.codecogs.com/svg.latex?d_{model})），那么结果变为：
```python
tensor([[ 0.4924,  0.3564,  0.4850],
        [ 1.0244, -0.0162, -0.0017],
        [ 0.6738,  0.7967,  0.6629]])
```
在这个3 x 3 的矩阵里，每一行代表一个词，对于改变后的序列，每一个词的位置（pos）分别为0, 1, 2。那么，公式的i则代表被嵌入的维度的索引，这里每一个词被嵌入到3维，拿字母"I"来说，第一行的每一个数分别对应于i = 0, 1, 2，这便是公式里的变量的意义。

接下来一步一步推演，首先我们假设公式里把2i去掉，这里换另一个比较长的序列，拿出第1，2，6，7个词，每个词用5维的向量表示，![](http://latex.codecogs.com/svg.latex?p_i)代表一个词

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/begin.png)

正弦图像是具有周期性的，不巧的话就会出现不同的位置是一个值，真是那样的话，位置编码的意义就失去了，如何让不同词之间分开是位置编码的目的，接下来引入i，依照公式，只有i为偶数才为正弦图，引入i之后代表进一步细分，不止是按照一个一个词来分，而是按照词和词嵌入的特征来分，其实可以想想，人其实大多数情况的行为举止都是差不多的，细分才能看出不同，这里用了![](http://latex.codecogs.com/svg.latex?p_0)和![](http://latex.codecogs.com/svg.latex?p_6) 第1，3，5个特征：

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/second.png)

不难发现，不仅不同词之间分开了，同一个词不同的特征之间也能区分出来

当两个参数都变得很大的时候，又会发生什么呢？

transformer默认的词嵌入维度是512，这里举个特例，假如句子很长，以第101个词为例

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/special.png)

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/sin.png)

显然，前期有大量特征会出现重合，且会出现较高频率变换，后期几乎趋于一致（奇数情况也如此），我们其实可以猜想出后面的其实几乎不带有位置信息，下图为奇偶混合的情况，可以看出有很多重合，当i较大时，两者都不带任何信息

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/mix.png)

下图为小demo的位置编码的可视化结果

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/positional_encoding.png)

深度是一个词的特征维度，Position代表的是哪一个词，可以看出，当i较小时，特征之间重合较大，而且离得近的位置之间特征变化很快，当i大到一定程度，几乎没有任何区别

对于上述现象，我的猜想是当词与词之间离得很近时（pos差得小）或者对于一个词不同特征来说，携带信息相同其实是符合逻辑的。在word2vec里面，如果两个词意相同，转换为坐标空间内的向量就会离得近（对应于一个词不同特征），有时候，一个句子之间离得近的词可能有关联效果，如"I love reading books."，reading和books它们一起出现的概率是较大的（对应于离得近的词）。

位置编码是加上去的，为什么不拼接上去呢？

拼接的话会扩大参数空间，占用内存增加，而且不易拟合，而且其实没有证据表明拼接就比加来的好

***

**<div id='add'>Add & Norm</div>**

Add 的意思是残差相连，思路来源于论文[Deep residual learning for image recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)，Norm指的是Layer Normalization，来源于论文[Layer normalization](https://arxiv.org/abs/1607.06450)。

论文中指出
> That is, the output of each sub-layer is
LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer
itself

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/add.jpg)


意思是上一层的输入和输出结果相拼接，然后进行归一化，这样可以更加稳定。肯定会有读者好奇为什么不直接将输出结果归一化，偏要把输入结果拼接到输出结果然后再归一化。

如果还有读者不明白Backprogation，建议看[BP](https://github.com/sherlcok314159/ML/blob/main/NN/bp.md)，这里简要说明一下，求后向传播的过程中，设残差相连之后输入的层为L层，那么，肯定要求这一层对残差相连的时候的偏导数，而这时x是作为自变量的，所以对于F(x)+ x，求偏导后x就会变为1，那么无论什么时候都会加上这个常数1，这样会一定程度上缓解梯度消失这个问题。

这里选取的是层归一化（Layer Normalization），用CNN举例，假如我们的输入是![](http://latex.codecogs.com/svg.latex?[N,C,H,W])，分别代表样本数量（每一个神经元对应着一个样本），通道数，高度和宽度，那么LN就是对于单独的每一个样本做归一化，计算![](http://latex.codecogs.com/svg.latex?C*H*W)个值的均值和标准差然后应用到这个样本里面。

归一化的公式如下：

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/ln.png)


***

**<div id='feed'>Feed Forward</div>**

Feed-Forward Network究竟做了啥呢？

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/FFN_2.png)

首先它会引入[RELU](../NN/activation.md)进行非线性变化，也就是公式前半部分所为，而且经过这一层之后会被升维，之后把这一层的结果连接到下一层进行线性变化，同时进行降维，保证输入输出维度一致。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/FFN.png)

***
**<div id='drop'>Residual Dropout</div>**

在此模型中还在输入，输出层和位置编码拼接中采用了dropout，这里的比例是0.1。

***
**<div id='etd'>Encoder To Decoder</div>**

接下来要把左侧的编码器和右侧的解码器进行相连，细心观察图会发现只有两个箭头到了右边，这两个箭头代表的是K，V矩阵，Q矩阵由右侧解码器提供。

另外，解码器会**MASK**掉序列的某些部分，因为如果要判断机器是否真正理解了整段句子的意思，可以把后面遮掉，它如果猜的出来，那么说明机器理解了。具体来说，对于位置i，通常会将i+1后面的全都**MASK**掉，这样机器就是从前面学到的，它没有提前看到后面的结果。


其他结构与编码器一致，最后再进行线性变化，最终用softmax映射成概率。
***
**<div id='share'>Shared Weights</div>**

我们需要把输入和输出转换为向量表示，而向量表示的方法是提前学到的。此外，最后用的线性转换以及softmax都是学出来的。和常规的序列模型一致，输入输出以及线性转换用的权重矩阵是共享的，只不过在输入输出层用的时候乘以模型维度开根号。
***
**<div id='effect'>Effect</div>**

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/effect.png)

一般来说，维度d肯定比序列长度n大很多，所以每一层的复杂度此模型吊打RNN。模型结果对比没多说的，几乎是碾压式的。

***
**<div id='references'>References</div>**

https://arxiv.org/abs/1706.03762

https://kazemnejad.com/blog/transformer_architecture_positional_encoding/

https://www.youtube.com/watch?v=dichIcUZfOw&t=3s