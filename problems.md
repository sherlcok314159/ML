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

- 终端运行bash文件

```bash
bash xxx.sh
```

- "Consider using the `--user` option or check the permissions."

```bash
pip3 install --user 包名
```

- bash运行报错，默认Linux执行bash是python2
```
python3 main.py
```

- pip批量安装

```bash
pip3 install --user -r requirements.txt
```

- pip全局设定清华源

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

临时使用：
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pip -U
```
***

- 检测 GPU运行时间和内存
```
%load_ext scalene
```

**<div id='file'>文件处理</div>**

- json文件结构不清楚，可以在vscode中格式化文档，用json美化拓展

- Ubuntu 直接解压zip文件会出现中文乱码现象，用下列命令：
```bash
unzip -O CP936 xxx.zip
```

- Ubuntu解压.rar文件

```
sudo apt-get install unrar
unrar e filename
```

- 生成该环境所有包

```bash
pip freeze > requirements.txt
```

- 生成某个项目的包

```
pip install pipreqs
cd 根目录
pipreqs ./ --encoding=utf8 --force
```

- 中文乱码问题

试试转换编码（UTF-8/GB-18030/GBK）

- 程序报错

利用try-except语句或者编辑器自带的debug工具

- Expected object of scalar type Float but got scalar type Double for argument #3 'weight' in call to _thnn_nll_loss_forward

```python
weight = np.ones(39,dtype=np.float32)
```

float32和float64（`FLOAT`）并不等价


- curl命令出现无法获取内容或需要权限

```bash
sudo curl url
```

- json.loads单引号报错，直接用
```python
js = json.loads(json.dumps(eval(s)))
```

- 引用包出错
```python
from pysrc.Config import *
```

- pandas读取文件

```python
data = pd.read_csv(path, header=None, sep="\t")
```
- pip 找不到

尝试更新Pip
```bash
pip install --upgrade pip
```

- 一个Pip源没有的时候
```bash
pip install --upgrade pip -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
```
- from_pretrained()报

```python
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```
不指定具体路径即可

```python
model = xxx.from_pretrained(dir)
```
- 写入json
```python
with open("test.json", "w) as f:
  json.dump("{}", f)
```

- latex 两列

```python
\begin{figure*}

\end{figure*}
```

- matplotlib保存图片不清晰
```python
plt.savefig('output.png', dpi=500, bbox_inches='tight')
plt.show() ## show()一定要在后面，否则会空白
```

- zsh 安装后conda不能正常使用

```python
vim ~/.zshrc

export PATH=/home/username/anaconda3/bin:$PATH

source ~/.zshrc
```

- latex 左引号(``)

- latex 索引排序

```python
\bibliographystyle{unsrt}
```
- latex bib排序

```python
1. 创建.bib文件
2. 开头：
\usepackage{cite}
\begin{document}
3. 
\bibliographystyle{unsrt/plain}
\bibliography{UN}
```

- latex表格宽度

```python
\setlength{\belowcaptionskip}{0.2cm}
```

- latex斜体公理
```python
\newtheorem{assump}{Assumption}

\begin{assump}
xxx
\end{assump}
```

- latex 共一

```python
\equalcontrib
```

- word打印图片吞图

```
插入而非复制图片
```

- ubuntu查看所有已安装包

```bash
apt list --installed
```
- ubuntu查看某个目录或包大小

```
du -sh directory/file
```

- ubuntu查看代理设置
```
env|grep -i proxy
```
若不能上网，直接将所有proxy置为零即可
```
export http_proxy=""
export https_proxy=""
export ftp_proxy=""
export NO_PROXY=""
```
