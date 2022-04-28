# neufarm
Tool for data preprocess in ML.
# Usage
## Algorithms
### Mean
```python
from Algorithms import Mean
d = {'col1': [0, 1, 2]}
df = pd.DataFrame(data=d)
mean = Mean(vector=df.col1)
Output: 1
```
### Median
```python
from Algorithms import Median
d = {'col1': [0, 1, 2, 3]}
df = pd.DataFrame(data=d)
median = Median(vector=df.col1)
Output: 1.5
```
### Mode
```python
from Algorithms import Mode
d = {'col1': [0, 1, 2, 3, 1]}
df = pd.DataFrame(data=d)
mode = Mode(vector=df.col1)
Output: 1
```
### Euclid metric
```python
from Algorithms import EuclidMertic
d = {'col1': [0, 1, 2], 'col2': [2, 1, 0]}
df = pd.DataFrame(data=d)
euqld  = EuclidMertic(vector1=df.col1, vector2=df.col2) 
Output: 2.8284271247461903
```
### Manhattan metric
```python
from Algorithms import ManhattanMetric
d = {'col1': [0, 1, 2], 'col2': [2, 1, 0]}
df = pd.DataFrame(data=d)
mnht = ManhattanMetric(vector1=df.col1, vector2=df.col2) 
Output: 4.0
```
### Max Metric
```python
from Algorithms import MaxMetric
d = {'col1': [0, 1, 2], 'col2': [2, 1, 0]}
df = pd.DataFrame(data=d)
mx = MaxMetric(vector1=df.col1, vector2=df.col2)
Output: 2
```
### Correlation Coefficient
```python
from Algorithms import CorrelationCoefficient
d = {'col1': [99, 89, 91, 91, 86, 97], 'col2': [58, 48, 54, 54, 44, 56]}
df = pd.DataFrame(data=d)
cc = CorrelationCoefficient(vector1=df.col1, vector2=df.col2)
Output: 0.906843948104356
```
## Rertore value methods
### You have to send df with NaN value. It is important that the NaN value be either above or below the table. 
Example:
```
   P1  P2  P3  P4   P5
0   3   4   5   3  4.0
1   5   5   5   4  3.0
2   4   3   3   2  5.0
3   5   4   3   3  NaN
```
Or
```
      G   T   B    R   W
0   NaN  56  91  160  58
1  89.0  58  89  157  48
2  91.0  64  91  165  54
3  91.0  51  91  170  54
4  86.0  56  84  157  44
5  97.0  53  86  175  56
6  92.0  51  91  165  54
7  87.0  55  88  170  53
8  91.0  55  90  165  55
```
```python
from RestoreValue import MetricRestore
d = {'P1': [3, 5, 4, 5], 'P2': [4, 5, 3, 4], 'P3': [5, 5, 3, 3], 'P4': [3, 4, 2, 3], 'P5': [4, 3, 5, np.NaN]}
df = pd.DataFrame(data=d)
# Euclid
euclid_m = MetricRestore(df, row_start=0, row_end=9, metric='euclid')
# Manhattan
mnht_m = MetricRestore(df, row_start=0, row_end=9, metric='manhattan')
# Max
mx_m = MetricRestore(df, row_start=0, row_end=9, metric='max')

Output: 
euclid_m = 4.13
mnht_m = 4.1
mx_m = 4.25
```