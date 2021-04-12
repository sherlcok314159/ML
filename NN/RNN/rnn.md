### Recurrent Neutral Network

章节

- [RNN概述](#summary)
- [传统困难](#problems)
- [LSTM](#lstm)
- [应用](#application)

### <div id='summary'>RNN概述</div>

为什么它叫做递归神经网络呢？与其他网络有何不同？接下来用简单例子阐述：

这是比较简单的示意图，比如说一个网络只有一层，那么，那一层代表的函数方法就是这个网络实际对输入所起的作用，即Y = Funtion(X)，我们实际上想找出那个function它究竟是什么。

![](https://github.com/sherlcok314159/ML/blob/main/Images/network.png)

可以从下图看出，RNN得到一个输出不仅仅靠输入的X，同时还依赖于h，h在RNN中被叫做cell state，那么h如何得出呢？由公式（1）可知，h_t是由h_(t-1)经过某种函数变换得到的，换句话说，我要得到目前这一个的，我还必须经过前一个才能做到。这里我们可以类比一下斐波那契数列，f(t) = f(t-1) + f(t-2)，某一项需要由前两项一起才能完成，RNN是某一个h需要前面一个h来完成，这也是为什么被叫做递归神经网络。顺带一提，这里的function有权重参数，即为W,而这个W是共享的，意思是无论是h_1到h2还是h_2到h_3，它们用的function其实是一样的。

![](https://github.com/sherlcok314159/ML/blob/main/Images/rnn.png)

![](https://github.com/sherlcok314159/ML/blob/main/Images/ht.png)

***
### <div id='problems'>传统困难</div>

1
***
### <div id='lstm'>LSTM</div>

1
***
### <div id='application'>应用</div>

1
***

