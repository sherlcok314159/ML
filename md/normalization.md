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

![](https://github.com/sherlcok314159/ML/blob/main/Images/min_max.png)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

data = np.array([[23, 140, 11, 340, 12], [12, 222, 353, 132, 23]], dtype=float)

max_ = np.max(data, axis=1, keepdims=True)
min_ = np.min(data, axis=1, keepdims=True)
data = (data - min_) / (max_ - min_)
sns.distplot(data, fit=stats.norm)
# plt.xlabel("Unchanged data")
plt.xlabel("Changed data")
plt.show()
```

这样处理之后会使整个数据分布在[0,1]，最大值会变为1，最小值会变为0，越大就会越接近1，越小就越接近0

整个数据不会因此变得符合正态分布

![](https://github.com/sherlcok314159/ML/blob/main/Images/min_max_unchanged.png)

![](https://github.com/sherlcok314159/ML/blob/main/Images/min_max_changed.png)

### Z-score

![](https://github.com/sherlcok314159/ML/blob/main/Images/mean_sigmoid.png)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

data = np.array([[23, 140, 11, 340, 12], [12, 222, 353, 132, 23]], dtype=float)
mean_ = np.mean(data, axis=1, keepdims=True)
std_ = np.std(data, axis=1, keepdims=True)
data = (data - mean_) / std_
print(data)
[[-0.64718872  0.27399231 -0.74166883  1.84866073 -0.73379549]
 [-1.06590348  0.57515027  1.59885522 -0.12815848 -0.97994352]]
```
处理后的数据，如果大于平均值就是正的，如果小于平均值则为负的

![](https://github.com/sherlcok314159/ML/blob/main/Images/min_max_unchanged.png)

![](https://github.com/sherlcok314159/ML/blob/main/Images/mean_sigmoid_changed.png)
