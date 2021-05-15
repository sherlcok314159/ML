# DL
此仓库将介绍Deep Learning 所需要的基础知识以及NLP方面的模型原理到项目实操 : )

### 章节
- [线上课程](#courses)
- [深度学习](#deep)
- [Transformer时代](#transformer)
- [必备技能](#skills)
- [问题库](#problems)
****
### <div id='courses'>线上课程</div>

- [李宏毅](https://www.bilibili.com/video/BV1JE411g7XF?from=search&seid=5913060628821893017)（入门极荐，简单易懂，且为中文授课）
- [陈蕴侬](https://www.bilibili.com/video/BV19g4y1b7vx?p=28&spm_id_from=pageDriver)（与李宏毅同为台大老师，偏重应用，范围更广）
- [吴恩达](https://www.bilibili.com/video/BV164411b7dx?from=search&seid=3957162850020779432)（斯坦福华人教授，业界公认厉害）
- [Hinton](https://www.bilibili.com/video/BV1Xf4y117nc?from=search&seid=2370955475461598590)（深度学习开山鼻祖）

****

### <div id='deep'>Neural Network</div>
- [Gradient Descent](optimization/GD.md)
- [BackPropagation](NN/bp.md)
- [Normalization](https://github.com/sherlcok314159/ML/blob/main/Book/NLP_Notes.pdf)
- [Activation](NN/activation.md)
- [DNN](NN/DNN/dnn.md)
    - [IMDB影评情感分类](NN/DNN/IMDB.md)
- [CNN](NN/CNN/cnn.md)
- [RNN概述](NN/RNN/rnn.md)
    - [单双向RNN](NN/RNN/单向rnn、双向rnn_embedding.py)
    - [LSTM之词嵌入](NN/RNN/LSTM/lstm_embedding.py)
    - [LSTM之文本分类](NN/RNN/LSTM/文本分类_lstm_subword.py)
    - [LSTM之文本生成](NN/RNN/LSTM/文本生成.py)
    - [Seq2Seq翻译模型搭建](NN/RNN/seq2seq.md)
- [常见优化技巧](NN/problems.md)

***
### <div id='transformer'>Transformer时代</div>

- [文本预处理](nlp/embedded.md)

- NLP模型原理讲解
    - [Transformer](nlp/models/transformer.md)
    - [Bert](nlp/models/bert.md)

- Bert文本任务源码解读系列
     - [Bert之文本分类](nlp/tasks/text.md)
     - [Bert之阅读理解](nlp/tasks/understand.md)

- Bert实战项目
     - [中文情感分类](nlp/practice/sentiment.md)

### <div id='skills'>必备技能</div>

- [阅读源码](nlp/source_code.md)
- [迅速上手项目](nlp/fast.md)
- [Numpy](data_process/numpy.md)

### <div id='problems'>问题库</div>

此部分用来记载各种各样小bug的解决方案

- [问题库](problems.md)

