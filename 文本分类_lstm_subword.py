# -*- coding: utf-8 -*-
"""文本分类-LSTM-subword.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yAuE8tx8tAEJ8auF6K5v95AnoDWvnApY
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import sklearn 
import os
import sys
import time
# 加载数据集特别常用
import tensorflow_datasets as tfds

print(tf.__version__)
print(sys.version_info)
for module in mpl,np,pd,sklearn,tf,keras,tfds:
  print(module.__name__,module.__version__)

"""https://tensorflow.google.cn/datasets/catalog/overview
好多数据集：音频、图片、问答、文本、翻译、视频、可视化
"""

# 影评分类

# 下载subword数据集
# with_info:返回元组（tf.data.Dataset，tfds.core.DatasetInfo）
# as_supervised True:有监督的，会把labels返回 False:无监督的，不会把labels返回
# info:subword形成的集合
dataset,info=tfds.load('imdb_reviews/subwords8k',with_info=True,as_supervised=True)
train_dataset,test_dataset=dataset['train'],dataset['test']

# 看看输入、输出
print(train_dataset)
print(test_dataset)

"""输入是(None,)
输出是()
"""

train_dataset = train_dataset.map(lambda x_text, x_label: (x_text, tf.expand_dims(x_label, -1)))
test_dataset = test_dataset.map(lambda x_text, x_label: (x_text, tf.expand_dims(x_label, -1)))

# encoder:把文本转成subword形式
# tokenizer对象
tokenizer=info.features['text'].encoder
print(type(tokenizer))
# 看看词袋里面有哪些单词
print('vocabulary size:{}'.format(tokenizer.vocab_size))

# 从训练集中拿出一个，看看有哪些词根subword
for i in train_dataset.take(1):
  print(i)

# 对于随便一个句子，看看它在词袋中的id

sample_string="Tensorflow is cool."
# encode():把文本变为subword的id序列
tokenized_string=tokenizer.encode(sample_string)
print("Tokenized string is {}".format(tokenized_string))
# decode():把subword的id序列变为文本
original_string=tokenizer.decode(tokenized_string)
print("Original string is {}".format(original_string))

assert original_string==sample_string

# 看看这个例子中的每个subword的id
for token in tokenized_string:
  print("{}—>{} len:{}".format(token,tokenizer.decode([token]),len(tokenizer.decode([token]))))

"""空格也有id"""

# 获取shape

buffer_size=10000
batch_size=64

padded_shapes=tf.compat.v1.data.get_output_shapes(train_dataset)
print(padded_shapes)
padded_shapes_test=tf.compat.v1.data.get_output_shapes(test_dataset)
print(padded_shapes_test)

train_dataset=train_dataset.shuffle(buffer_size)
print(train_dataset)

# padded_batch()对每批数据做padding
train_dataset_=train_dataset.padded_batch(batch_size,padded_shapes)
test_dataset=test_dataset.padded_batch(batch_size,padded_shapes_test)
print(train_dataset)
print(test_dataset)

"""batch之后维度增加了

"""

vocab_size=tokenizer.vocab_size
embedding_dim=16
batch_size=512

# 双向单层LSTM
bi_lstm_model=keras.models.Sequential([
                           keras.layers.Embedding(vocab_size,embedding_dim),
                           keras.layers.Bidirectional(keras.layers.LSTM(units=32,return_sequences=False)),
                           keras.layers.Dense(32,activation='relu'),
                           keras.layers.Dense(1,activation='sigmoid')                               
])

bi_lstm_model.summary()
bi_lstm_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

"""subword 词袋大小：8185
8185x16=130960
"""

history=bi_lstm_model.fit(train_dataset,epochs=10,validation_data=test_dataset)

def plot_learning_curves(history,label,epochs,min_value,max_value):
  data={}
  data[label]=history.history[label]
  data['val_'+label]=history.history['val_'+label]
  pd.DataFrame(data).plot(figsize=(8,5))
  plt.grid(False)
  plt.axis([0,epochs,min_value,max_value])
  plt.show()

plot_learning_curves(history,'accuracy',10,0,1)
plot_learning_curves(history,'loss',10,0,1)

"""在验证集上：accuracy效果好，loss也没有过拟合，subword-level效果最好啊！！！"""