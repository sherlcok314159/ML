### 问题库

集锦

- Colab
  - [深度学习环境](#deep)



**<div id='deep'>Colab——深度学习</div>**

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

