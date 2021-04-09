# ML
此仓库将介绍Machine Learning 所需要的基础知识 : )

### 章节
- [线上课程](#courses)
- [基础知识](#basic)
- [深度学习](#deep)
- [NLP](#nlp)
- [CV](#cv)
- [必备技能](#skills)
- [问题库](#problems)
****
### <div id='courses'>线上课程</div>

- [李宏毅](https://www.youtube.com/watch?v=CXgbekl66jc&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=1)（入门极荐，简单易懂，且为中文授课）
- [吴恩达](https://www.coursera.org/learn/machine-learning/home/welcome)（斯坦福华人教授，业界公认厉害）

****
### <div id='basic'>基础知识</div>
<!-- **** -->
数据处理
* [Normalization](data_process/normalization.md)

模型汇总
* [Regression](models/Regression/04_training_linear_models.ipynb)

损失函数
* [Loss function](loss/loss_.md)

优化方法
* [Gradient Descent](optimization/GD.md)
****
其实神经网络属于机器学习，只是过程具体化有些不同罢了，所以分开处理

### <div id='deep'>Neutral Network</div>
- [Activation](NN/activation.md)
- [DNN](NN/DNN/dnn.md)
    - [Sentiment Classification in IMDB](NN/DNN/IMDB.md)
- [CNN](NN/CNN/cnn.md)
- [RNN](NN/RNN/单向rnn、双向rnn_embedding.py)
    - [LSTM之词嵌入](NN/RNN/LSTM/lstm_embedding.py)
    - [LSTM之文本分类](NN/RNN/LSTM/文本分类_lstm_subword.py)
    - [LSTM之文本生成](NN/RNN/LSTM/文本生成.py)
- [优化](NN/problems.md)

### <div id='nlp'>NLP任务</div>

- NLP模型原理讲解
    - [Transformer](nlp/models/transformer.md)
    - [Bert](nlp/models/bert.md)

- Bert文本任务源码解读系列
     - [Bert之文本分类](nlp/tasks/text.md)
     - [Bert之阅读理解](nlp/tasks/understand.md)

- Bert实战项目
     - [中文情感分类](nlp/practice/sentiment.md)

### <div id='cv'>CV</div>

### <div id='skills'>必备技能</div>

- [阅读源码](nlp/source_code.md)
- [迅速上手项目](nlp/fast.md)
- [Numpy](data_process/numpy.md)

### <div id='problems'>问题库</div>

此部分用来记载各种各样小bug的解决方案

- [问题库](problems.md)

