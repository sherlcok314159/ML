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
data_dir --> 数据集的位置
vocab_dir --> 词表的位置（一般bert模型下好就能找到） 
bert_config --> bert模型参数设置
init_checkpoint --> 预训练好的模型
max_seq_length --> 一个序列的最大长度
output_dir --> 结果输出文件（包括日志文件）
其他的字面意思
```