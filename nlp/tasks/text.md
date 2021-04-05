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
        - [词向量编码](#token)
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

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/label_map.png)

label_list前面对数据进行处理的类里有get_labels参数，返回的是一个列表，如["0","1"]。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/load_vocab.png)

想要切分数据，首先得读取词表吧，代码里面一开始创造一个OrderedDict，这个是为什么呢？

在python 3.5的时候，当你想要遍历键值对的时候它是任意返回的，换句话说它并不关心键值对的储存顺序，而只是跟踪键和值的关联程度，会出现**无序**情况。而OrderedDict可以解决无序情况，它内部维护着一个根据插入顺序排序的双向链表，另外，对一个已经存在的键的重复复制不会改变键的顺序。

需要注意，OrderedDict的大小为一般字典的两倍，尤其当储存的东西大了起来的时候，需要慎重权衡。

但是到了python 3.6，字典已经就变成有序的了，为什么还用OrderedDict，我就有些疑惑了。如果说OrderedDict排序用得到，可是普通dict也能胜任，为什么非要用OrderedDict呢？


![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/or.png)


在tokenization.py文件中提供了三种切分，分别是BasicTokenizer，WordpieceTokenizer和FullTokenizer，下面具体介绍一下这三者。

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

以上便是BasicTokenizer的内容了。

接下来是WordpieceTokenizer了，其实这个词切分是针对英文单词的，因为汉字每个字已经是最小的结构，不能进行切分了。而英文还可以进行切分，英文有不同语态，如loved,loves,loving等等，这个时候WordpieceTokenizer就能发挥作用了。

- 遍历一个英文单词里面的小结构，如果发现在词表里找到，就把这个切掉
- 对未被切分的部分继续进行步骤一，直至所有都被切分干净，注意除了第一个，其他的前面都要加上"##"

下面有个gif可以直观显示，[来源](https://alanlee.fun/2019/10/16/bert-tokenizer/)

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/v2-4771d1cfbda7282b74e5713e628290f0_b.gif)


最后是FullTokenizer，这个是两者的集成版，先进行BasicTokenizer，后进行WordpieceTokenizer。当然了，对于中文，就没必要跑WordpieceTokenizer。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/full.png)

下面简单提一下convert_by_vocab，这里是将具体的内容转换为索引。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/id.png)

以上就是切分了。
***

**<div id='embedding'>词向量编码</div>**

刚刚对数据进行了切分，接下来我们跳到函数convert_single_example，进一步进行词向量编码。


![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/single_features.png)

这里是初始化一个例子。input_ids 是等会把一个一个词转换为词表的索引；segment_ids代表是前一句话（0）还是后一句话（1），因为这还未实例化，所以is_real_example为false。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/tok.png)

此处tokenizer.tokenize是FullTokenizer的方法。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/b_none.png)

不同的任务可能含有的句子不一样，上面代码的意思就是若b不为空，那么max_length = 总长度 - 3，原因注释已有；若b为空，则就需要减去2即可。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/truncate.png)

_truncate_seq_pair进行一个截断操作，里面用了pop()，这个是列表方法，把列表最后一个取出来，英文注释也说了为什么没有按照比例截断，若一个序列很短，那按比例截断会流失信息较多，因为比例是长短序列通用的。同时，_truncate_seq_pair还保证了a,b长度一致。若b为空，a则不需要调用这个方法，直接列表方法取就好。


我们不是说需要在开头添加[CLS]，句子分割处和结尾添加[SEP]嘛（本次任务a,b均不为空），刚刚只是进行了一个切分和截断操作。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/cls.png)

tokens是我们用来放序列转换为编码的新列表，segment_ids用来区别是第一句还是第二句。这段代码大意就是在开头和结尾处加入[CLS]，[SEP]，因为是a所以都是第一句，segment_ids就都为0，同时[CLS]和[SEP]也都被当做是a的部分，编码为0。下面关于b的同理。

接下来再把具体内容转换为索引。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/ids.png)

我们一开始的参数不是有max_seq_length嘛，这个代表一整个序列的最大长度（a,b拼接的），但是很多时候我们的总序列长度不会达到最大长度，但是我们又要保证所有输入序列长度一致，即为最大序列长度。所以我们需要对剩下的部分，即没有内容的部分进行填充（Padding），但填充的时候有个问题，一般我们都会添0，但做self-attention的时候（如果还不了解自注意力，可以去主页看看我写的Transformer的论文解读），每一个词要跟句子里面所有的词做内积，但是0是我们人为填充进去的，它不代表任何意义，然而，做自注意力的时候还是要跟它做内积，是不是不太合理呀？

于是就有了MASK机制，什么意思呢？我们把机器需要看，需要做自注意力的保留，不要看的MASK掉，这样做自注意力的时候就不会出岔子。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/mask.png)

同时，只要没达到最大长度，就全部补零。

这个的剩余部分tf.logging是日志，不用管，这个convert_single_example最终返回的是feature，feature包含什么已经具体阐述过了。


****

**<div id='tf'>TFRecord文件构建</div>**


因为用TFRecord读取文件比较方便快捷，需要转换一下文件格式。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/example.png)

前半部分是examples写入，examples是来自上图方法。features是来自上面刚讲过的convert_single_example方法。

需要注意的是这份run_classifier.py人家谷歌是用TPU跑的，所以会有TPU部分代码，一般我们只用GPU，所以TPU部分不需要关注，一般TPU都会出现TPUEstimator。

***

**<div id='model'>模型构建</div>**

接下来，是构建模型篇，是整个代码中最重要的一部分。接下来我将用代码介绍一下transformer模型的架构。

找到modeling.py文件，这是模型文件。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/bertconfig.png)

首先是BertConfig的类，这里自定义了一些参数及数值。

```python
vocab_size --> 词表的大小，用别人的词表，这个参数已经固定
hidden_size --> 
num_hidden_layers --> 全连接层隐含层的层数
num_attention_heads -->注意力头的个数
intermediate_size --> 全连接层中隐含层层数
hidden_act --> 全连接层中的激活函数
hidden_dropout_prob --> 在全连接层中实施Dropout，被去掉的概率
attention_probs_dropout_prob --> 
max_position_embeddings --> 最大位置数目
type_vocab_size -->
initializer_range -->
```

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/shape.png)

后面 batch_size x seq_length 会经常出现，这里是原始定义

这里还有个初始化，如果MASK和token_type_ids我们前面没有，这里就默认全为1和0。这是为了后面词嵌入（embedding）做准备。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/mask_type.png)

***

**<div id='connect'>词向量拼接</div>**

接下来正式进入Embedding层的操作，最终传到注意力层的其实是原始token_ids，token_type_ids以及positional embedding拼接起来的。

**<div id='token'>token_ids编码</div>**

首先是token_ids的操作，先来看一下embedding_lookup方法。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/embedding_lookup.png)

这是它的参数，大部分英文注释已有，需要注意的一点是input_ids的shape必须为[batch_size,max_seq_length]。

接下来进行扩维。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/input_shape.png)

等会我们需要在embedding_table里面查找，这里先构建一个[vocab_size,embedding_size]的table。需要注意的是vocab_size 和 embedding_size 都是固定好的，训练的时候不能乱改。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/embedding_table.png)

之后我们对input_ids进行降维，貌似这样可以加速。one_hot_embedding一般为false，这是对TPU加速用的。接下来在embedding_table里面进行查找。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/gather.png)

然后我们把output reshape一下。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/output_reshape.png)

这就是token的编码了。

***

**<div id='type'>句子类型编码</div>**

进行位置编码之前，我们首先进行对token_type_ids的编码（判断是哪一句）。

首先创建token_type_table。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/token_type_table.png)

然后进行一个token_type_embedding，matul是矩阵相乘

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/type_embedding.png)

做好相乘之后，我们需要把token_type_embedding的shape还原，因为等会要将token_type_ids与词编码相加。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/type_reshape.png)

***
**<div id='position'>位置编码</div>**

首先我们先创造大量的位置，max_position_embeddings是官方给定的参数，不能修改。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/full_position.png)

我们创造了这么多的位置，最终不一定用的完，为了更快速的训练，我们一般做切片处理，只要到我的max_seq_length还有位置就好，后面都可以不要。


![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/slice.png)

前面要把token_type_embeddings加到input_ids的编码中，进行了同维度处理，这里对于位置编码也一样，不然最后相加不了。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/position_reshape.png)

至此，Embedding层就结束了。

***

