### Bert & Transformer源码详解

参考论文

https://arxiv.org/abs/1706.03762

https://arxiv.org/abs/1810.04805


在本文中，我将以run_classifier.py以及MRPC数据集为例介绍关于bert以及transformer的源码，另外，本文在最后一个部分详细讲解了如何从0到1来跑自己的第一个bert模型。

**章节**
- [Demo传参](#flags)
    - [跑不动?](#bugs)
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

首先大家拿到这个模型，管他什么原理，肯定想跑起来看看结果，至于预训练模型以及数据集下载。任何时候应该先看[官方教程](https://github.com/google-research/bert)，官方代表着权威，更容易实现，如果遇到问题可以去issues和stackoverflow看看，再辅以中文教程，一般上手就不难了，这里就不再赘述了。

先从Flags参数讲起，到如何跑通demo。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/flags.png)

拿到源码不要慌张，英文注释往往起着最关键的作用，另外阅读源码详细技巧可以看[源码技巧](../source_code.md)。

"Required Parameters"意思是必要参数，你等会执行时必须向程序里面传的参数。

```bash
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=/path/to/glue

python run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/tmp/mrpc_output/
```

这是官方给的示例，这个将两个文件夹加入了系统路径，本人Ubuntu18.04加了好像也找不到，所以建议**将那些文件路径改为绝对路径**。

```python
task_name --> 这次任务的名称
do_train --> 是否做fine-tune
do_eval --> 是否交叉验证
do_predict --> 是否做预测
data_dir --> 数据集的位置
vocab_dir --> 词表的位置（一般bert模型下好就能找到） 
bert_config --> bert模型参数设置
init_checkpoint --> 预训练好的模型
max_seq_length --> 一个序列的最大长度
output_dir --> 结果输出文件（包括日志文件）
do_lower_case --> 是否小写处理（针对英文）

其他的字面意思
```
***
**<div id='bugs'>跑不动？</div>**

有些时候发现跑demo的时候会出现各种问题，这里简单汇总一下

**1.No such file or directory!**

这个意思是没找到，你需要确保你上面模型和数据文件的路径填正确就可解决

**2.Memory Limit**


![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/bug.png)

因为bert参数量巨大，模型复杂，如果GPU显存不够是带不动的，就会出现上图的情形不断跳出。

解决方法

- 把batch_size,max_seq_length,num_epochs改小一点
- 把do_train直接false掉
- 使用优化bert模型，如Albert,FastTransformer

经过本人实证，把参数适当改小参数，如果还是不行直接不做fine-tune就好，这对迅速跑通demo的人来说最有效。

***

**<div id='data'>数据篇</div>**

这是很多时候我们自己跑别的任务最为重要的一章，因为很多时候模型并不需要你大改，人家都已经给你训练好了，你在它的基础上进行优化就好了。而数据如何读入以及进行处理，让模型可以训练是至关重要的一步。

***

**<div id='read'>数据读入</div>**

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/data.png)


简单介绍一下我们的数据，第一列为Quality，意思是前后两个句子能不能匹配得起来，如果可以即为1，反之为0。第二，三两列为ID，没什么意义，最后两列分别代表两个句子。

接下来我们看到DataProcessor类，（有些类的作用仅仅是初始化参数，本文不作讲解）。这个类是父类（超类），后面不同任务数据处理类都会继承自它。它里面定义了一个读取tsv文件的方法。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/read.png)

首先会将每一列的内容读取到一个列表里面，然后将每一行的内容作为一个小列表作为元素加到大列表里面。

***


