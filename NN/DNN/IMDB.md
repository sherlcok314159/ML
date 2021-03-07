### Sentiment Classification in IMDB

***通过DNN实现影评情感分类***

章节

- [包的准备](#prepare)
- [数据下载](#download)
- [文本表示](#txt_)
- [数据预处理](#preprocess)
- [设计DNN](#design)
- [其他方案](#w)


**<div id='prepare'>包的准备</div>**

```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

for module in mpl,np,pd,tf,keras:
    print(module.__name__,module.__version__)

'''
matplotlib 3.3.4
numpy 1.16.0
pandas 1.1.5
tensorflow 1.14.0
tensorflow.python.keras.api._v1.keras 2.2.4-tf
'''

若没有按照版本安装，如
pip3 install numpy==1.16.0
```

**<div id='download'>数据下载</div>**

```python
#取出词频为前10000
vocab_size = 10000
# <3的id都是特殊字符
index_from = 3
# 这里用keras里面的imdb数据集
imdb = keras.datasets.imdb

(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words = vocab_size，index_from = index_from)

# 若要进一步了解各种参数，API示例文档 https://keras.io/api/datasets/imdb/

# 训练集的大小
print(train_data.shape)
print(train_labels.shape)

# 训练集的第一个样本（是向量）
print(train_data[0],train_labels[0])

# 多维的数组 numpy.ndarray
print(type(train_data))
print(type(train_labels))

# train_labels的值0(negative),1(positive)-二分类
print(np.unique(train_labels))

'''
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz
17465344/17464789 [==============================] - 28s 2us/step
(25000,)
(25000,)
[1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515, 17, 12, 16, 626, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 5952, 15, 256, 4, 2, 7, 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2, 1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16, 5345, 19, 178, 32] 1
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
[0 1]
'''

```

