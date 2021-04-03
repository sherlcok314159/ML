### Bidirectional Encoder Representations from Transformers


参考论文(https://arxiv.org/abs/1810.04805)

**章节**
- [Encoder](#encoder)
- [Ways Of Training](#train)
    - [MASK](#mask)
    - [Connect Or Not](#connect)
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


[CLS]代表分类的结果，对于一个二分类任务就是可以或不可以连在一起（0，1）

