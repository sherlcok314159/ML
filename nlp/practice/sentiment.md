### 中文情感分类单标签

**章节**
- [背景介绍](#bg)
- [预处理](#preprocess)
- [模型优化](#better)


**<div id='bg'>背景介绍</div>**

这次的任务是中文的一个评论情感去向分类

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/data_.png)

每一行一共有三个部分，第一个是索引，无所谓；第二个是评论具体内容；第三个是标签，由0，1，2组成，1代表很好，2是负面评论，0应该是情感取向中立。

***

**<div id='preprocess'>数据预处理</div>**

bert模型是可以通用的，但是不同数据需要通过预处理来达到满足bert输入的标准才行。

首先，我们创造一个读入自己数据的类，名为MyDataProcessor。其实，这个可以借鉴一下谷歌写好的例子，比如说MrpcProcessor。

首先将DataProcessor类复制粘贴一下，然后命名为MyDataProcessor，别忘了继承一下DataProcessor。

接下来我们以get_train_examples为例来简单介绍一下如何读入自己的数据。

第一步我们需要读取文件进来，这里需要注意的是中文要额外加一个utf-8编码。

![](https://github.com/sherlcok314159/ML/blob/main/nlp/Images/file.png)

读取好之后，这里模仿创建train_data为空列表，索引值为0。

代码主体跟其他的差不多，