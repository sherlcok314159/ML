### Recurrent Neutral Network

章节

- [RNN概述](#summary)
- [LSTM](#lstm)
- [梯度困区](#problems)
- [应用](#application)

### <div id='summary'>RNN概述</div>

为什么它叫做递归神经网络呢？与其他网络有何不同？接下来用简单例子阐述：

这是比较简单的示意图，比如说一个网络只有一层，那么，那一层代表的函数方法就是这个网络实际对输入所起的作用，即Y = Funtion(X)，我们实际上想找出那个function它究竟是什么。

![](https://github.com/sherlcok314159/ML/blob/main/Images/network.png)

可以从下图看出，RNN得到一个输出不仅仅靠输入的X，同时还依赖于h，h在RNN中被叫做cell state，那么h如何得出呢？由公式（1）可知，h_t是由h_(t-1)经过某种函数变换得到的，换句话说，我要得到目前这一个的，我还必须经过前一个才能做到。这里我们可以类比一下斐波那契数列，f(t) = f(t-1) + f(t-2)，某一项需要由前两项一起才能完成，RNN是某一个h需要前面一个h来完成，这也是为什么被叫做递归神经网络。顺带一提，这里的function有权重参数，即为W,而这个W是共享的，意思是无论是h_1到h2还是h_2到h_3，它们用的function其实是一样的。

![](https://github.com/sherlcok314159/ML/blob/main/Images/rnn.png)

![](https://github.com/sherlcok314159/ML/blob/main/Images/ht.png)

所以，复杂一点的RNN长这样：

![](https://github.com/sherlcok314159/ML/blob/main/Images/rnn_1.png)

每次输出完一个y，它同时还会有一个h出来，作为下一层的参数一起使用。从这一点来看，RNN跟其他网络不同的一点是前一层的输出同时可以作为后一层的输入，经过一层就会更新一次h，那么，h究竟是如何更新的呢？tanh是一种常用的激活函数，可见[Activation](../activation.md)。

![](https://github.com/sherlcok314159/ML/blob/main/Images/htt.png)

y_t可以由此得出：

![](https://github.com/sherlcok314159/ML/blob/main/Images/y_t.png)

从上述公式中可以看出有不同的W，即不同的权重矩阵，这些矩阵是机器自己去从数据中去学出来，同时也可以是人为设置的。注意，这些不同类之间的矩阵不同，但是如果说是同一个function，那么权重矩阵都是共享的。

传统的DNN，CNN的输入和输出都是固定的向量，而RNN与这些网络的最大不同点是它的输入和输出都是不定长的，具体因不同任务而定。

![](https://github.com/sherlcok314159/ML/blob/main/Images/many_lengths.png)

***
### <div id='lstm'>LSTM</div>

LSTM和GRU比较有创新的一点就是采用了门结构来控制整个模型，既然是门，那就可以打开和关闭，如何定义打开还是关闭呢？我们用sigmoid来完成这一点，如果经过sigmoid函数的值越接近0，受到重视的程度就越低，相当于门正在慢慢关闭，越接近于1呢，受到重视的程度就越高，相当于门正在慢慢打开，下面把LSTM切分为不同的门结构来讲。

![](https://github.com/sherlcok314159/ML/blob/main/Images/final_.png)

我相信你一开始看到这个图是一脸懵逼的，接下来我带你手撕LSTM

- Forget Gate

![](https://github.com/sherlcok314159/ML/blob/main/Images/f_t.png)

其中两个W都是权重矩阵，两个b都是截距，是通过机器去不断学出来的，下文出现的W和b虽然具体内容不同，但是代表的意思是一样的。忘记门决定了哪些信息是重要的，如果是不重要的我们就直接选择遗忘，是LSTM中较为核心的一点。

![](https://github.com/sherlcok314159/ML/blob/main/Images/f_t_.png)

- Input Gate

![](https://github.com/sherlcok314159/ML/blob/main/Images/i_t.png)

![](https://github.com/sherlcok314159/ML/blob/main/Images/i_t_.png)

- Cell Gate

![](https://github.com/sherlcok314159/ML/blob/main/Images/g_t.png)

需要注意的是，这里的激活函数换成了tanh。

![](https://github.com/sherlcok314159/ML/blob/main/Images/g_t_.png)

- Cell State

![](https://github.com/sherlcok314159/ML/blob/main/Images/c_t.png)

![](https://github.com/sherlcok314159/ML/blob/main/Images/c_t_.png)

式子分为两部分，前一部分是说前面的cell state有哪些需要保留，哪些需要遗忘，cell gate用来暂存需要补充到新的c_t的内容。两者相加，便完成了cell state的更新了。其实为什么叫cell state——细胞状态，在我看来，不妨从细胞膜的选择透过性来说，这里c_t的更新不是上一部分直接拿上来就用，而是进行选择性录入，跟物质运送到细胞内有异曲同工之妙。

同时你会发现整个LSTM很大一部分都是围绕着Cell State展开的，那些门间接在保护或者过滤输出，至于为什么LSTM能缓解梯度消失以及维持一个较为稳定的梯度流，可以在[梯度困区](#problems)中找到答案，下一节会具体比对RNN和LSTM。

注意这里的是哈达玛积（Hadamard product），是对应位置元素相乘。

![](https://github.com/sherlcok314159/ML/blob/main/Images/had_.png)


- Output Gate

![](https://github.com/sherlcok314159/ML/blob/main/Images/o_t__.png)

![](https://github.com/sherlcok314159/ML/blob/main/Images/o_t_.png)


- Hidden State

![](https://github.com/sherlcok314159/ML/blob/main/Images/h_t.png)

![](https://github.com/sherlcok314159/ML/blob/main/Images/h_t_.png)

最后我们完成LSTM一层的搭建

![](https://github.com/sherlcok314159/ML/blob/main/Images/final.png)

叠加三层就长成了一开始的样子：

![](https://github.com/sherlcok314159/ML/blob/main/Images/final_.png)

梯度消失导致RNN只能捕获到比较近的信息，也就是tanh压缩之后的信息，而丧失了远距离传过来的信息，导致它并不能处理很长的句子，LSTM只是缓解并没有解决这一问题，说起LSTM的名字也很有趣，Long Short-Term Network，其实只是比较长的短期网络啦。
***

### <div id='problems'>梯度困区</div>

RNN通过Hidden State（h_t）路径完成梯度流动：

![](https://github.com/sherlcok314159/ML/blob/main/Images/rnn_ht.png)

由上式易得，权重矩阵和激活函数很容易对RNN的梯度造成不可逆的影响，关于sigmoid函数为什么会导致梯度下降问题，建议去[BackPropagation](../bp.md)中的梯度消失部分一看究竟，这里不再细说。其实最重要的不是激活函数，对梯度传播真正起决定作用的是权重矩阵，因为随着梯度的传播过程，乘以权重矩阵的指数倍，换句话说，若权重矩阵里都是比较小的数，那么，梯度就会指数性下降；同样地，如果权重矩阵里都是比较大的数，那么，梯度就会指数性上升。

所以，对RNN梯度下降以及梯度爆炸问题，可以从这两个角度进行切入。

- 梯度初始化

我们可以初始化权重矩阵使之变为正交矩阵，最简单的初始方法就是使权重矩阵变为单位阵（Identity Matrix），这样随着梯度不断的流动，可以缓解指数性上升或者下降的问题。

- 切换激活函数

因为sigmoid会导致梯度下降，所以我们可以切换激活函数如RELU或者RELU的变种，如Leaky RELU。

另外针对梯度爆炸问题，可以采用梯度削减（Gradient Clipping）：

首先设置一个clip_gradient作为梯度阈值，然后按照往常一样求出各个梯度，不一样的是，我们没有立马进行更新，而是求出这些梯度的L2范数，注意这里的L2范数与岭回归中的L2惩罚项不一样，前者求平方和之后开根号而后者不需要开根号。如果L2范数大于设置好的clip_gradient，则求clip_gradient除以L2范数，然后把除好的结果乘上原来的梯度完成更新。当梯度很大的时候，作为分母的结果就会很小，那么乘上原来的梯度，整个值就会变小，从而可以有效地控制梯度的范围。有一点疑惑的就是，梯度削减会使得原来的梯度过大的部分发生变化，方向既然发生了变化，为什么最后还能使得loss收敛呢？Deep Learning大概结果反推出解释吧。

当然了，上面这些措施只能是稍作改变，不痛不痒，为了更好地缓解这些问题，LSTM被提了出来，结构已经介绍过了，其实LSTM绝对不能解决上述梯度问题，最多进行缓解，它可以在一条路径上保持较为稳定的梯度流——Cell State（c_t），其他的路径上同样会有梯度消失的问题，因而不能说解决。

![](https://github.com/sherlcok314159/ML/blob/main/Images/c_t_gd.png)


- 升级模型

在RNN的基础上提出了RNN的变种——GRU和LSTM，下面的部分来解析LSTM和GRU。
***
### <div id='application'>应用</div>

1
***

