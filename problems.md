### 问题库

集锦

- [Colab](#colab)
- [处理文件](#file)


**<div id='colab'>Colab——深度学习</div>**

- 切换tensorflow版本：
```python
%tensorflow_version 1.x #切换至1.x
import tensorflow
print(tensorflow.__version__) #验证版本
```

- 下载东西：
```bash
!wget + 链接 （colab命令行前面加!，另外，下载东西可以一起下，中间加空格即可）
# e.g. !wget https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt

```

- 源码项目：
```bash
!git clone + 链接
# e.g. !git clone https://github.com/carlos9310/bert.git
```

- 创造文件夹
```bash
!mkdir SQUAD_DIR && cd SQUAD_DIR # 导航至该文件下
# 另外可以支持多条命令 &&即可 
```

- 切换GPU：> 代码执行程序 > 更改运行类型

- 计时

```bash
%%time
```

- debug
```bash
%debug print(i)
```
![](https://github.com/sherlcok314159/ML/blob/main/Images/debug.png)

- 定位到自己云盘

```python
from google.colab import drive
drive.mount("/content/drive")
```

- 把云盘某文件夹当做工作区

```bash
!cd /content/drive/MyDrive/R-transformer
```

```python
import os
path = "/content/drive/MyDrive/R-transformer"
os.chdir(path)
os.listdir(path)
```

- 终端运行Python文件

```bash
python file.py
```

- bash报错

bash不能留空

***

- 检测 GPU运行时间和内存
```
%load_ext scalene
```

**<div id='file'>文件处理</div>**

- json文件结构不清楚，可以在vscode中格式化文档，用json美化拓展