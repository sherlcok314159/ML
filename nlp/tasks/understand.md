### Bert & Transformer阅读理解源码详解

参考论文

https://arxiv.org/abs/1706.03762

https://arxiv.org/abs/1810.04805


在本文中，我将以run_squad.py以及SQuAD数据集为例介绍阅读理解的源码，官方代码基于tensorflow-gpu 1.x，若为tensorflow 2.x版本，会有各种错误，建议切换版本至1.14。

当然，注释好的源代码在[这里](https://github.com/sherlcok314159/ML/tree/main/nlp/code)

**章节**
- [Demo传参](#flags)
- [数据篇](#data)
  - [番外句子分类](#outside)
  - [创造实例](#example)
  - [实例转换](#convert)

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

前面的参数都十分常规，如果不懂，建议看我的[文本分类](text.md)的讲解。这里讲一下比较特殊的最后一个参数，我们做的任务是阅读理解，如果有答案缺失，在SQuAD1.0是不可以的，但是在SQuAD允许，这也就是True的意思。

需要注意，不同人的文件路径都是不一样的，你不能照搬我的，要改成自己的路径。

***


**<div id='data'>数据篇</div>**

其实阅读理解任务模型是跟文本分类几乎是一样的，大的差异在于两者对于数据的处理，所以本篇文章重点在于如何将原生的数据转换为阅读理解任务所能接受的数据，至于模型构造篇，请看[文本分类](text.md)。

***

**<div id='outside'>番外句子分类</div>**

想必很多人看到SquadExample类的_repr_方法都很疑惑，这里处理好一个example，为什么后面还要进行处理？看英文注释会发现这个类其实跟阅读理解没关系，它只是处理之后对于句子分类任务的，自然在run_squad.py里面没被调用。

_repr_方法只是在有start_position的时候进行字符串的拼接。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/repr.png)

***

**<div id='example'>创造实例</div>**

用于训练的数据集是json文件，需要用json库读入。

训练集的样式如下，可见data是最外层的

```python
{
    "data": [
        {
            "title": "University_of_Notre_Dame",
            "paragraphs": [
                {
                    "context": "Architecturally, the school has a Catholic character.",
                    "qas": [
                        {
                            "answers": [
                                {
                                    "answer_start": 515,
                                    "text": "Saint Bernadette Soubirous"
                                }
                            ],
                            "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
                            "id": "5733be284776f41900661182"
                        }
                    ]
                }
            ]
        },
        {
            "title":"...",
            "paragraphs":[
                {
                    "context":"...",
                    "qas":[
                        {
                            "answers":[
                                {
                                    "answer_start":..,
                                    "text":"...",
                                }
                            ],
                            "question":"...",
                            "id":"..."
                        },
                    ]
                }
            ]
        }
    ]
}
```

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/json.png)

input_data是一个大列表，然后每一个元素样式如下

```python
{'paragraphs': [{...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, ...], 'title': 'University_of_Notre_Dame'}
```

is_whitespace方法是用来判断是否是一个空格，在切分字符然后加入doc_tokens会用到。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/white.png)


然后我们层层剥开，然后遍历context的内容，它是一个字符串，所以遍历的时候会遍历每一个字母，字符会被进行判断，如果是空格，则加入doc_tokens，char_to_word_offset表示切分后的索引列表，每一个元素表示一个词有几个字符组成。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/context.png)

切分后的doc_tokens会去掉空白部分，同时会包括英文逗号。一个单词会有很多字符，每个字符对应的索引会存在char_to_word_offset，例如，前面都是0，代表这些字符都是第一个单词的，所以都是0，换句话说就是第一个单词很长。

```python
doc_tokens = ['Architecturally,', 'the', 'school', 'has', 'a', 'Catholic', 'character.', 'Atop', 'the',"..."]

char_to_word_offset = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...]
```
接下来进行qas内容的遍历，每个元素称为qa，进行id和question内容的分配，后面都是初始化一些参数

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/qas.png)

qa里面还有一个is_impossible，用于判断是否有答案

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/ispossible.png)

确保有答案之后，刚刚读入了问题，现在读入与答案相关的部分，读入的时候注意start_position和end_position是相对于doc_tokens的

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/answer.png)

接下来对答案部分进行双重检验，actual_text是根据doc_tokens和始末位置拼接好的内容，然后对orig_answer_text进行空格切分，最后用find方法判断orig_answer_text是否被包含在actual_text里面。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/find.png)

这个是针对is_impossible来说的，如果没有答案，则把始末位置全部变成-1。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/impossible.png)

然后将example变成SquadExample的实例化对象，将example加入大列表——examples并返回，至此实例创建完成。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/example_.png)

***

**<div id='convert'>实例转换</div>**

把json文件变成实例之后，我们还差一步便可以把数据塞进模型进行训练了，那就是将实例转化为变量。

先对question_text进行简单的空格切分变为query_tokens

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/example_tokenize.png)

如果问题过长，就进行截断操作

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/question_slice.png)

接下来对doc_tokens进行空格切分以及词切分，变成all_doc_tokens，需要注意的是orig_to_tok_index是all_doc_tokens的索引

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/subtoken.png)

对tok_start_position和tok_end_position进行初始化，记住，这两个是相对于all_doc_tokens来说的，一定要与start_position和end_position区分开来，它们是相对于doc_tokens来说的

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/tok_start.png)

接下来先介绍_improve_answer_span方法，这个方法是用来处理特殊的情况的，举个例子，假如说你的文本是"The Japanese electronics industry is the lagest in the world."，你的问题是"What country is the top exporter of electornics?" 那答案其实应该是Japan，可是呢，你用空格和词切分的时候会发现Japanese已经在词表中可查，这意味着不会对它进行再切分，会直接将它返回，这种情况下可能需要这个方法救场。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/_improve.png)

因为是监督学习，答案已经给出，所以呢，这个方法干的事情就是词切分后的tokens进行再一次切分，如果发现切分之后会有更好的答案，就返回新的始末点，否则就返回原来的。

对tok_start_position和tok_end_position进行进一步赋值

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/tok_.png)


计算max_tokens_for_doc，与文本分类类似，需要减去[CLS]和两个[SEP]的位置，这里不同的是还要减去问题的长度，因为这里算的是文本的长度。

tokens = [CLS] query tokens [SEP] context [SEP]

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/squad_length.png)