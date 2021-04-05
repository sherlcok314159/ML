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
- [词处理](#deal)
    - [切分](#split)
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
**<div id='handle'>数据处理</div>**


因为我们的数据集为MRPC，我们直接跳到MrpcProcessor类就好，它是继承自DataProcessor。

这里简要介绍一下os.path.join。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/os_path.png)

我们不是一共有三个数据集，train,dev以及test嘛，data_dir我们给的是它们的父目录，我们如何能读取到它们呢？以train为例，是不是得"path/train.tsv"，这个时候，os.path.join就可以把两者拼接起来。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/labels.png)

这个意思是任务的标签，我们的任务是二分类，自然为0&1。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/read_.png)

examples最终是列表，第一个元素为列表，内容图中已有。

***
**<div id='deal'>词处理</div>**
读取数据之后，接下来我们需要对词进行切分以及简单的编码处理

***


**<div id='split'>切分</div>**

刚刚对数据进行了简单的处理，接下来我们调到函数convert_single_example，进一步进行词向量编码。


![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/single_features.png)

这里是初始化一个例子。input_ids 是等会把一个一个词转换为词表的索引；segment_ids代表是前一句话（0）还是后一句话（1），因为这还未实例化，所以is_real_example为false。


![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/label_map.png)

label_list前面对数据进行处理的类里有get_labels参数，返回的是一个列表，如["0","1"]。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/load_vocab.png)

想要切分数据，首先得读取词表吧，代码里面一开始创造一个OrderedDict，这个是为什么呢？

在python 3.5的时候，当你想要遍历键值对的时候它是任意返回的，换句话说它并不关心键值对的储存顺序，而只是跟踪键和值的关联程度，会出现**无序**情况。而OrderedDict可以解决无序情况，它内部维护着一个根据插入顺序排序的双向链表，另外，对一个已经存在的键的重复复制不会改变键的顺序。

需要注意，OrderedDict的大小为一般字典的两倍，尤其当储存的东西大了起来的时候，需要慎重权衡。

但是到了python 3.6，字典已经就变成有序的了，为什么还用OrderedDict，我就有些疑惑了。如果说OrderedDict排序用得到，可是普通dict也能胜任，为什么非要用OrderedDict呢？


![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/or.png)


在tokenization.py文件中提供了三种分类，分别是BasicTokenizer，WordpieceTokenizer和FullTokenizer，下面具体介绍一下这三者。

在tokenization.py文件中遍布convert_to_unicode，这是用来转换为unicode编码，一般来说，输入输出不会有变化。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/clean.png)

这个方法是用来替换不合法字符以及多余的空格，比如\t,\n会被替换为两个标准空格。

接下来会有一个_tokenize_chinese_chars方法，这个是对中文进行编码，我们首先要判断一下是否是中文字符吧，_is_chinese_char方法会进行一个判断。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/chinese.png)

如果是中文字符，_tokenize_chinese_chars会将中文字符旁边都加上空格，图中我也有引例注释。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/whitespace.png)

whitespace_tokenize会进行按空格切分。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/accents.png)

_run_strip_accents会将变音字符替换掉，如résumé中的é会被替换为e。

接下来进行标点字符切分，前提是判断是否是标点吧，_is_punctuation履行了这个职责，这里不再多说。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/punc.png)


![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/v2-4771d1cfbda7282b74e5713e628290f0_b.gif)