### 词嵌入

章节

- [词嵌入概述](#summary)
- [skip-gram](#skip)
- [CBOW](#cbow)
- [Negative Sample](#ns)
- [GloVe](#glove)
- [ELMo](#elmo)
- [bert](#bert)
- [参考文献](#references)

### <div id='summary'>词嵌入概述</div>

深度学习任务中，我们不能直接将词语送入模型，我们需要将其转换为数值矩阵。常见的问题是如何以较低维度的矩阵来表示词从而减少运输量；在转换为数值之后仍然保留不同词之间的相关性；同词不同场景的情况如何表示；如何使得出现频率不同的词都能得到较好的训练等等

这篇词嵌入会具体梳理前辈中做的工作以及目前的主流操作，以原理为主，因大多数已渐被取代且预训练不易（需要大量数据集），不切入代码实践

***

### <div id='skip'>skip-gram</div>

先介绍one-hot（独热编码）为什么在NLP中不大能胜任，我们不仅需要将词转换为矩阵，而且还要保持不同词之间的关系，比如余弦相似度：

![](https://github.com/sherlcok314159/ML/blob/main/pre/Images/cos.png)

而不同的单词用one-hot做内积之后结果均为0，这样就会丧失词之间的关系

skip-gram就是选出中心词来预测其他词出现在它周围的概率，例如，一个句子是"the man loves his car."，假设"loves"是中心词，引入一个context window的概念，即为周围两侧覆盖的范围，若为2，那么左侧的"the man"和右侧"his car"都会被覆盖到。P(the,man,his,car|loves)意为当中心词为"loves"，那么在context window范围内，它周围词为这些的概率

假设![](http://latex.codecogs.com/svg.latex?w_i,w_c)分别代表context word和central word（文本词与中心词）以及![](http://latex.codecogs.com/svg.latex?u_i,v_c)代表它们所被表示成的向量，![](http://latex.codecogs.com/svg.latex?\mathcal{V})代表词表内单词的数量，那么给定中心词，任意文本词作为它邻居的概率其实是softmax：

![](https://github.com/sherlcok314159/ML/blob/main/pre/Images/skip.png)

那么给定一个长度为T，t为每一个时间步，m是context window的大小，那么，将所有概率相乘，并且每一个词都可以作为文本词和中心词，这就意味着每一个词有二维向量，分别对应不同的场景，即为：

![](https://github.com/sherlcok314159/ML/blob/main/pre/Images/skip_.png)

例如，句子长度为5，m为2，句子仍为"the man loves his car"，在第一个时间步时：

![](https://github.com/sherlcok314159/ML/blob/main/pre/Images/skip_t1.png)

对于时间步小于1和大于T的不予考虑，另外对自身不做softmax概率，那么![](http://latex.codecogs.com/svg.latex?P(w^2|w^1),P(w^3|w^1))分别代表man，loves从the中生成的概率，在不同时间步上中心词都不同。

极大似然概率被用于训练skip-gram，SGD常用于skip-gram的参数更新

![](https://github.com/sherlcok314159/ML/blob/main/pre/Images/skip_max.png)

联系上面![](http://latex.codecogs.com/svg.latex?P(w_i|w_c))的定义（注意其实w的上标和下标并没有本质区别，上标只是为了更清楚地表示时间步而已）可以得到：

![](https://github.com/sherlcok314159/ML/blob/main/pre/Images/skip_log.png)

接下来我们求![](http://latex.codecogs.com/svg.latex?v_c)的梯度（其实分子和分母的下标应该是一致的，这里处理不是为了分子分母同除，为了区分，所以采用不同下标）：

![](https://github.com/sherlcok314159/ML/blob/main/pre/Images/skip_log_.png)


***

### <div id='cbow'>CBOW</div>

其实CBOW和skip-gram最大的不同是前者是由中心词产生周围邻居，而CBOW恰恰相反，由周围词产生中心词。继续上面的例子，"the man loves his car"，那么对应的中心词概率为P(loves|the,man,his,car)

因为中心词数量过多，这里平均处理，相对应的softmax概率即为：

![](https://github.com/sherlcok314159/ML/blob/main/pre/Images/cbow.png)

为了简便，记![](http://latex.codecogs.com/svg.latex?\mathcal{W}_0=\\{w_{o1},\dots,w_{o2m}\\})，
![](http://latex.codecogs.com/svg.latex?\bar{\mathbf{v}}_o=(v_{o1}+\dots+v_{o2m})/(2m))

所以上面的式子简化为

![](https://github.com/sherlcok314159/ML/blob/main/pre/Images/cbow_simple.png)

所以，在给定时间步长T下（同skip-gram）：

![](https://github.com/sherlcok314159/ML/blob/main/pre/Images/cbow_.png)

那么，极大似然概率为

![](https://github.com/sherlcok314159/ML/blob/main/pre/Images/cbow_log.png)

联系上面的式子，进行简化：

![](https://github.com/sherlcok314159/ML/blob/main/pre/Images/cbow_concrete.png)

计算![](http://latex.codecogs.com/svg.latex?\bar{\mathbf{v}}_{oi})的梯度

![](https://github.com/sherlcok314159/ML/blob/main/pre/Images/cbow_last.png)

不难发现，当需要求一个参数的梯度时，skip-gram和CBOW都需要将整个词表乘一遍，当词表很大的时候，计算会非常耗时

***

### <div id='ns'>Negative Sample</div>

为了解决skip-gram和CBOW都会遍历词表，复杂度为![](http://latex.codecogs.com/svg.latex?\mathbf{O}(n))，一个方法是hierarchical softmax，它是通过哈夫曼树让复杂度降至![](http://latex.codecogs.com/svg.latex?\mathbf{O}(logn))，但较为复杂而且也不是普遍应用，这里忽略，详细介绍另一种方法Negative Sampling，下方简写为NS

NS其实是符合直觉的，一开始是遍历整个词表，那么有没有可能遍历从词表中取样出来的小样本呢？不断学习，不就间接上等于把词表整个遍历了吗？

这里用skip-gram为例，CBOW和它差不多，假设给定一个句子，![](http://latex.codecogs.com/svg.latex?w_c,w_o)分别代表中心词和句子中出现的词，![](http://latex.codecogs.com/svg.latex?v_c,u_o)分别代表它们被表示成的向量，P(D=1)代表正样本，即为![](http://latex.codecogs.com/svg.latex?w_o)出现在中心词的context window里

![](https://github.com/sherlcok314159/ML/blob/main/pre/Images/ns.png)

那么，在给定时间步上，![](http://latex.codecogs.com/svg.latex?w^t)代表不同时间步时的中心词

![](https://github.com/sherlcok314159/ML/blob/main/pre/Images/ns_prod.png)

我们取对数处理

![](https://github.com/sherlcok314159/ML/blob/main/pre/Images/ns_prod_log.png)

意味着我们希望不同的中心词时，邻居出现在它context window的可能性尽可能大，还是"the man loves his car"，这次假设context window大小为1，那么当t=1时，则希望在所有词中"man"出现在它旁边的概率最大，以此类推，t=2，则希望"the"，"loves"出现在"man"周围的概率最大等等


负采样的意思是不仅要让所有在context window里的正样本概率变大，同时从不在window里的词中采样![](http://latex.codecogs.com/svg.latex?\mathcal{K})个词作为噪声词，然后把两个样本概率相乘，记![](http://latex.codecogs.com/svg.latex?P(w))为负样本采样时的分布，![](http://latex.codecogs.com/svg.latex?w_k)为噪声词

![](https://github.com/sherlcok314159/ML/blob/main/pre/Images/ns_joint.png)

正负样本符合伯努利分布，即两个相加为1，改写并取对数，极大似然概率如下：

![](https://github.com/sherlcok314159/ML/blob/main/pre/Images/ns_last.png)

问题来了，![](http://latex.codecogs.com/svg.latex?P(w))是多少呢？其实取样会有个问题，高频词被取样出来的概率会大于低频词，所以就希望能够平衡高频词和低频词之间的比例，原论文提出将每一个词的频率的3/4次方作为每一个词的词频，关于为啥是3/4，这是经验之谈，是通过实验结果找出的，并没有好的理论依托，不过可以通过例子来直观感受

假如一个句子中，loves，his，metal出现的概率分别为0.9，0.09，0.01（即该词出现的次数/总词数），那么变换之后分别是0.92，0.16，0.032，可以发现虽然每一个值都上升了，但是高频词上升的程度远没有低配词来的大，这样就可以一定程度平衡比例
****

### <div id='glove'>GloVe</div>

经过上述介绍，不难发现skip-gram和CBOW都是通过滑动窗口来捕捉词与词之间的关系，但是没有用到整个序列的统计信息，而且需要用到大量语料进行训练，训练较慢，而如LSA，HAL这类虽然可以较好地用到整个序列的统计信息，训练也较快，但是只能捕捉较为原始的词与词之间的关系。

Glove的出现结合了这两者的优点

记![](http://latex.codecogs.com/svg.latex?x_{ij})为![](http://latex.codecogs.com/svg.latex?x_i)出现在以![](http://latex.codecogs.com/svg.latex?x_j)为中心词的窗口中的概率，![](http://latex.codecogs.com/svg.latex?u_i,v_j)为文本词和中心词被表示成的向量，其实GloVe用的平方损失![](http://latex.codecogs.com/svg.latex?loss=(log\,x_{ij}-log\,u_i^Tv_j)^2)

