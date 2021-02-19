## 数据归一化

常见的数据归一化：**constant factor normalization,min-max,z-score**


### Constant Factor Normalization

最简单的归一化方法，将所有数据除以一个常数即可

假如我们的data有x1,x2,x3,x4，每个x有两个feature
```python
import numpy as np

data = np.array([[23, 140, 11, 340, 12], [12, 222, 353, 132, 23]], dtype=float)
data *= 0.01
print(data)

[[0.23 1.4  0.11 3.4  0.12]
 [0.12 2.22 3.53 1.32 0.23]]
```

### Min-Max

![](https://github.com/sherlcok314159/ML/blob/main/Images/eeaead2149745b81f52ea0ba53f75f5.png)
