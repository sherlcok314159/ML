### Bidirectional Encoder Representations from Transformers


参考论文(https://arxiv.org/abs/1810.04805)

**章节**
- [Encoder](#encoder)
- [Ways Of Training](#train)
    - [MASK](#mask)
    - [Connect Or Not](#connect)
- [Reading Comprehension](#comprehension)
- [Source Code Explanation](#code)


**<div id='encoder'>Encoder</div>**


其实Bert就是[Transformer](../nlp/transformer.md)编码器的部分。比较神奇的一点是并不需要标签，有预料即可训练


**<div id='train'>Ways Of Training</div>**

一共有两种方法训练Bert，一是MASK，二是拼接法。

**<div id='mask'>MASK</div>**

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/MASK.png)


我们可以定义一个概率决定多少的数据被遮盖掉，比如说设置为15%，然后我们让bert去猜遮掉的究竟是什么。bert会根据前后句子关系，然后去自己的语料库里找匹配的词语，然后返回最佳匹配的。英文一般均为一个英文单词，但是中文大多为一个字，**因为中文的词语简直太多，没有办法全部涵盖**

**<div id='connect'>Connect Or Not</div>**

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/connect.png)

------------------------------------------------------------------------[图片来源](https://www.bilibili.com/video/BV1NJ411o7u3?p=11)----------------------------------------------------------------------------


[CLS]代表分类的结果，对于一个二分类任务就是可以或不可以连在一起（0，1），[SEP]是句子连接符。这个任务就是判断两个句子是否可以合在一起，通过[CLS]输出结果。

***

**<div id='comprehension'>Reading Comprehension</div>**


比如说一篇文章里有这样一句话，Apples fall under gravity.那么问题可以就是What causes apples to fll? 答案为gravity。对于一篇文章，我们可以将整篇文章转换为向量表达，记为d1,d2,...,dn，同样的，问题可以记为q1,q2,...,qn，最终答案为qs,...,qe，答案是个序列，代表从哪开始，到哪结束。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/comprehension.png)

------------------------------------------------------------------------[图片来源](https://www.bilibili.com/video/BV1NJ411o7u3?p=11)----------------------------------------------------------------------------

这里需要额外训练两个辅助向量，因为答案我们需要始末位置，一个是开始，一个是结束。然后拿辅助向量与文章内容做内积，经过softmax得到最终概率，然后输出最相近的作为起始点，同样的得到结束点，然后答案就是始末位置卡住的地方。


***

**<div id='code'>Source Code Explanation</div>**

接下来我会通过对bert源码的解释来更加细致地了解bert的架构。