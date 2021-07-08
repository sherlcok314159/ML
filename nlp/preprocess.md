实战中数据往往不会像教程中的那样已经处理好，直接套模型就好，其实往往在数据的处理以及加载上花较多的时间，本文介绍如何预处理文本数据。

目录

- [去除\n, \t, \r](#clean)
- []


### <div id='clean'>去除\n, \t, \r</div>

原始的文本控制符：（`"\n"`），制表符（`\t`）和回车（`\r`）以及一些非法字符会影响模型对数据的读入，因而需要去除。


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
