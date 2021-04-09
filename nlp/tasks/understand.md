### Bert & Transformer阅读理解源码详解

参考论文

https://arxiv.org/abs/1706.03762

https://arxiv.org/abs/1810.04805


在本文中，我将以run_squad.py以及SQuAD数据集为例介绍阅读理解的源码，官方代码基于tensorflow-gpu 1.x，若为tensorflow 2.x版本，会有各种错误，建议切换版本至1.14。

当然，注释好的源代码在[这里](https://github.com/sherlcok314159/ML/tree/main/nlp/code)

**章节**
- [Demo传参](#flags)
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

```bash
python bert/run_squad.py \
  --vocab_file=uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=uncased_L-12_H-768_A-12/bert_model.ckpt \
  --do_train=True \
  --train_file=SQUAD_DIR/train-v2.0.json \
  --train_batch_size=8 \
  --learning_rate=3e-5 \
  --num_train_epochs=1.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=/tmp/squad2.0_base/ \
  --version_2_with_negative=True
```

阅读源码最重要的一点不是拿到就读，而是跑通源码里面的小demo，因为你跑通demo就意味着你对代码的一些基础逻辑和参数有了一定的了解。

前面的参数都十分常规，如果不懂，建议看我的[文本分类](text.md)的讲解。这里讲一下比较特殊的最后一个参数，我们做的任务是阅读理解，如果有答案缺失，在SQuAD1.0是不可以的，但是在SQuAD允许。这也就是True的意思。

