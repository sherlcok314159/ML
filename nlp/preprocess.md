实战中数据往往不会像教程中的那样已经处理好，直接套模型就好，其实往往在数据的处理以及加载上花较多的时间，本文介绍在送往模型之前，如何处理文本数据以及如何在训练过程中读入。

**目录**

- [读取数据](#load)
- [清洗数据](#clean)
- [Dataset对象](#dataset)
- [DataLoader](#dataloader)

### <div id='load'>读取数据</div>

- 读取csv/tsv

首先，选择的csv文件视图如下，以`\t`作为分隔符：

![](https://github.com/sherlcok314159/ML/blob/main/Images/example.png)

可以读取csv的方法大致分为`open()`和`pandas.read_csv()`两种，下面介绍两种方法的优点：

一般用`open`会与`csv.reader()`组合起来，每一行对应一个小列表，最终用大列表储存，好处是方便遍历

```python
def read_csv(your_file):
    with open(your_file, "r", encoding="utf-8") as f:
        return list(csv.reader(f, delimiter="\t"))

Example:

for line in read_csv(your_file):
    print(line)
    print(line[2])
    break

['id', 'title', 'body', 'category', 'doctype']
body
```

第二种是通过pandas进行读取，读取后是DataFrame对象

```python
import pandas as pd
train = pd.read_csv(your_file, sep="\t", encoding="utf-8")

print(type(train))
<class 'pandas.core.frame.DataFrame'>
print(train.head())
# 打印文件前几行
print(train.describe())
# 对各列数据进行统计
```

![](https://github.com/sherlcok314159/ML/blob/main/Images/show.png)

若想访问某一列的值，索引就是列名：

```python
print(train["doctype"])
```

读取某一列特定位置的值，`.values`是全部的值

```python
train["doctype"].values[2]
```

而且这个可以直接用来matplotlib画图，如：

```python
import matplotlib.pyplot as plt
plt.hist(train["doctype"])
plt.show()
```

同时，你还可以对于特定列的特定值进行过滤与替换

```python
# 定义新的一列new，为doctype列每一个值加1得到
train["new"] = train["doctype"].apply(lambda x : x+ 1)
# 选取doctype等于0的
train.loc[train["doctype"] == 0]
```

- 读取json文件

这里介绍json文件的读取，每一行为一个大字典，每一个字典的组成如下：

`{"id":xx, "title":xx, "body":xx, "category":xx, "doctype":xx}`

```python
from pandas import DataFrame
def read_json(your_file):
    data = [json.loads(line) for line in open(your_file, "r", encoding="utf-8")]
    data_ = DataFrame(data)
    data_.to_csv("example.csv", index=False, sep="\t")
```

***
### <div id='clean'>清洗数据</div>

原始的文本控制符：（`\n`），制表符（`\t`）和回车（`\r`）以及一些非法字符会影响模型对数据的读入，因而需要去除。


```python
def clean_text(text:str) -> str:
    '''去除非法字符以及控制字符 "\n", "\t", "r"'''
    output = []
    for char in text:
      cp = ord(char)
      if cp == 0 or cp == 0xfffd or is_control(char):
        continue
      if is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)


def is_control(char:str) -> bool:
  """检查是否为控制字符"""

  if char == "\t" or char == "\n" or char == "\r":
    return False
  cat = unicodedata.category(char)
  if cat in ("Cc", "Cf"):
    return True
  return False


def is_whitespace(char:str)-> bool:
  """检查是否为空白字符"""

  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False

Example:

string = "今天学文本处理很开心\n明天要去学校开会\t回家得好好学习\r"
string = clean_text(string)
print(string)

"今天学文本处理很开心 明天要去学校开会 回家得好好学习"
```
***
### <div id='dataset'>Dataset对象</div>

假如一开始的csv用来做多文本分类，`doctype`为标签，需要将`title`和`body`拼接起来作为`text_a`，`text_b`为`None`。一般`__init__`是将文件读取进来，也可以加点预处理步骤。`__len__`就是单纯返回数据集长度。`__getitem`有一个参数`idx`代表每次的索引，这个函数的作用是每次迭代时返回需要的信息。

```python
from torch.utils.data import Dataset
import pandas as pd
import csv

def read_csv(your_file):
    with open(your_file, "r", encoding="utf-8") as f:
        return list(csv.reader(f, delimiter="\t"))

# 自定义自己的数据集
class MyDataset(Dataset):
    def __init__(self, path):
        self.file = read_csv(path)

    def __len__(self):
        return len(self.file)
    
    def __getitem__(self, idx):
        guid = self.file[idx][0]
        text_a = self.file[idx][1] + self.file[idx][2]
        text_b = None
        label = self.file[idx][-1]
        return guid, text_a, text_b, label
```
***
### <div id='dataloader'>DataLoader</div>

```python
from torch.utils.data import DataLoader
from tqdm import tqdm

train_set = MyDataset(your_file)
train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)

# 方便计时
for batch in tqdm(train_dataloader, desc="Training"):
    model.train()
    guid, text_a, text_b, label = batch[0], batch[1], batch[2], batch[3]
    # 后续操作

```


