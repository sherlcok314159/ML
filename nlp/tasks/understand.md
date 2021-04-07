### Bert & Transformer阅读理解源码详解

参考论文

https://arxiv.org/abs/1706.03762

https://arxiv.org/abs/1810.04805


在本文中，我将以run_squad.py以及SQuAD数据集为例介绍阅读理解的源码，官方代码基于tensorflow-gpu 1.x，若为tensorflow 2.x版本，会有各种错误，建议切换版本至1.14。

当然，注释好的源代码在[这里](https://github.com/sherlcok314159/ML/tree/main/nlp/code)

**章节**
- [Demo传参](#flags)
    - [跑不动?](#bugs)
- [数据篇](#data)
    - [数据读入](#read)
    - [数据处理](#handle)
- [词处理](#deal)
    - [切分](#split)
    - [词向量编码](#embedding)
- [TFRecord文件构建](#tf)
- [模型构建](#model)
    - [词向量拼接](#connect)
        - [词向量编码](#token)
        - [句子类型编码](#type)
        - [位置编码](#position)
    - [多头注意力](#head)
        - [MASK机制](#mask)
        - [Q,K,V矩阵构建](#qkv)
    - [损失优化](#loss)
    - [其他注意点](#others)


**<div id='flags'>Demo传参</div>**