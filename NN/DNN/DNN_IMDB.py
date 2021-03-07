import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

for module in mpl,np,pd,tf,keras:
    print(module.__name__,module.__version__)

#取出词频为前10000
vocab_size = 10000
# <3的id都是特殊字符
index_from = 3
# 这里用keras里面的imdb数据集
imdb = keras.datasets.imdb

(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words = vocab_size,index_from = index_from)

print(train_data.shape)
print(train_labels.shape)

print(train_data[0],train_labels[0])

print(type(train_data))
print(type(train_labels))

print(np.unique(train_labels))

print(test_data.shape)
print(test_labels.shape)

word_index=imdb.get_word_index()

print(len(word_index))
print(type(word_index))

print(word_index.get("footwork"))

# 取出的词表索引从1开始，id都偏移3
word_index = {k:(v+3) for k,v in word_index.items()}
# 自定义索引0-3
word_index["<PAD>"] = 0 # 填充字符
word_index["<START>"] = 1 # 起始字符
word_index["<UNK>"] = 2 # 找不到就返回UNK
word_index["<END>"] = 3 # 结束字符

# 转成习惯的方式，索引为key,单词为value
reverse_word_index = {v:k for k,v in word_index.items()}

# 查看解码效果
print(reverse_word_index)

# {34710: 'fawn', 52015: 'tsukino', 52016: 'nunnery', 16825: 'sonja', 63960: 'vani', 1417: 'woods', ……}

# debug

print(reverse_word_index[34707])

# footwork

print(word_index.get("footwork"))

# 34707

# 随机看一下样本长度，可见长度不一
print(len(train_data[0]),len(train_data[1]),len(train_data[100]))

# 218 189 158

# 设置输入词汇表的长度，长度<500会被补全，>500会被截断

max_length = 500

# 填充padding
# value 用什么值填充
# padding 选择填充的顺序，2中pre,post
train_data = keras.preprocessing.sequence.pad_sequences(train_data,value = word_index["<PAD>"],padding="pre",maxlen = max_length)

# 使测试集要和训练集结构相同
test_data = keras.preprocessing.sequence.pad_sequences(test_data,value = word_index["<PAD>"],padding = "pre",maxlen = max_length)

# 一个单词的维度是16维
embedding_dim = 16
batch_size = 128

# 定义模型
# 定义矩阵 [vocab_size,embedding_dim]
# GlobalAveragePooling1D 全局平均值池化-在max_length这个维度上做平均，就是1x16了,在哪个维度上做Global，该维度就会消失
# 二分类问题，最后的激活函数用sigmoid
model = keras.models.Sequential([
        keras.layers.Embedding(vocab_size,embedding_dim,input_length = max_length),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(64,activation = "relu"),
        keras.layers.Dense(1,activation = "sigmoid"),
])

model.summary()
model.compile(optimizer = "adam",loss = "binary_crossentropy",metrics = ["accuracy"])

# 数据集中只有训练集、测试集，没有验证集，就用validation_split-拿20%的训练集数据当作验证集数据
history=model.fit(train_data,train_labels,epochs=30,batch_size=batch_size,validation_split=0.2)

# 绘制学习曲线
def plot_(history,label):
    plt.plot(history.history[label])
    plt.plot(history.history["val_" + label])
    plt.title("model " + label)
    plt.ylabel(label)
    plt.xlabel("epoch")
    plt.legend(["train","validation"],loc = "upper left")
    plt.show()

plot_(history,"acc")
plot_(history,"loss")

score = model.evaluate(
    test_data,test_labels,
    batch_size=batch_size,
    verbose=1
)

print("Test loss:", score[0])
print("Test accuracy:", score[1])

'''
Test loss: 0.6628
Test accuracy: 0.8588
'''
‵