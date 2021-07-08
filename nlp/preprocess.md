实战中数据往往不会像教程中的那样已经处理好，直接套模型就好，其实往往在数据的处理以及加载上花较多的时间，本文介绍如何预处理文本数据。

目录

- [去除\n, \t, \r](#clean)
- []


### <div id='clean'>去除\n, \t, \r</div>

原始的文本数据换行符（`"\n"`），制表符（`\t`）和回车（`\r`）会影响模型对数据的读入，因而需要去除。

```python
def no_blank(string:str) -> str:
    return string.replace("\n", " ").replace("\t", " ").replace("\r", " ")

Example:

string = "今天学文本处理很开心\n明天要去学校开会\t回家得好好学习\r"
string = no_blank(string)
print(string)

"今天学文本处理很开心 明天要去学校开会 回家得好好学习"
```

