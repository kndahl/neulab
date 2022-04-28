# neufarm
Tool for data preprocess in ML.
# Installation
```
pip install neulib
```
# Usage
## Algorithms
### Mean
```python
from neulib.Algorithms import Mean

d = {'col1': [0, 1, 2]}
df = pd.DataFrame(data=d)

mean = Mean(vector=df.col1)

Output: 1
```
### Median
```python
from neulib.Algorithms import Median
d = {'col1': [0, 1, 2, 3]}

df = pd.DataFrame(data=d)
median = Median(vector=df.col1)

Output: 1.5
```
### Mode
```python
from neulib.Algorithms import Mode

d = {'col1': [0, 1, 2, 3, 1]}
df = pd.DataFrame(data=d)

mode = Mode(vector=df.col1)

Output: 1
```
### Euclid metric
```python
from neulib.Algorithms import EuclidMertic

d = {'col1': [0, 1, 2], 'col2': [2, 1, 0]}
df = pd.DataFrame(data=d)

euqld  = EuclidMertic(vector1=df.col1, vector2=df.col2) 

Output: 2.8284271247461903
```
### Manhattan metric
```python
from neulib.Algorithms import ManhattanMetric

d = {'col1': [0, 1, 2], 'col2': [2, 1, 0]}
df = pd.DataFrame(data=d)

mnht = ManhattanMetric(vector1=df.col1, vector2=df.col2) 

Output: 4.0
```
### Max Metric
```python
from neulib.Algorithms import MaxMetric

d = {'col1': [0, 1, 2], 'col2': [2, 1, 0]}
df = pd.DataFrame(data=d)

mx = MaxMetric(vector1=df.col1, vector2=df.col2)

Output: 2
```
### Correlation Coefficient
```python
from neulib.Algorithms import CorrelationCoefficient

d = {'col1': [99, 89, 91, 91, 86, 97], 'col2': [58, 48, 54, 54, 44, 56]}
df = pd.DataFrame(data=d)

cc = CorrelationCoefficient(vector1=df.col1, vector2=df.col2)

Output: 0.906843948104356
```
## Rertore value methods
### You have to send df with NaN value. It is important that there is only one NaN in the table..
Example:
```
   P1  P2  P3  P4   P5
0   3   4   5   3  4.0
1   5   5   5   4  3.0
2   4   3   3   2  5.0
3   5   4   3   3  NaN
```
### Recovery with metrics
```python
from neulib.RestoreValue import MetricRestore

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
### Recovery with correlation coefficient
```python
from neulib.RestoreValue import CorrCoefRestore

d = {'G': [99, 89, 91, 91, 86, 97, np.NaN], 'T': [56, 58, 64, 51, 56, 53, 51], 'B': [91, 89, 91, 91, 84, 86, 91], 'R': [160, 157, 165, 170, 157, 175, 165], 'W': [58, 48, 54, 54, 44, 56, 54]}
df = pd.DataFrame(data=d)

cc = CorrCoefRestore(df=df, row_start=0, row_end=9)

Output: 94.21
```