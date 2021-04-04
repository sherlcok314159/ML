### Bert & Transformer源码详解

参考论文

https://arxiv.org/abs/1706.03762

https://arxiv.org/abs/1810.04805


在本文中，我将以run_classifier.py以及MRPC数据集为例介绍关于bert以及transformer的源码，另外，本文在最后一个部分详细讲解了如何从0到1来跑自己的第一个bert模型。

**章节**
- [Demo传参](#flags)
- [数据篇](#data)
    - [数据读入](#read)
    - [数据处理](#handle)
- [词向量编码](#embedding)
- [TFRecord文件构建](#tf)
- [模型构建](#model)
    - [词向量拼接](#connect)
        - [句子类型编码](#type)
        - [位置编码](#position)
    - [多头注意力](#head)
        - [MASK机制](#mask)
        - [Q,K,V矩阵构建](#qkv)


**<div id='flags'>Demo传参</div>**

首先大家拿到这个模型，管他什么原理，肯定想跑起来看看结果，至于预训练模型以及数据集下载，教程有很多，我就不赘述了。这里从Flags参数讲起，到如何跑通demo。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/flags.png)
