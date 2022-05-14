# neulab
Tool for data preprocess in ML.
# Installation
```
pip install neulab
```
# Usage

## Algorithms
### Mean
#### Algorithm for calculating the average value
```python
from neulab.Algorithms import Mean

d = {'col1': [0, 1, 2]}
df = pd.DataFrame(data=d)

mean = Mean(vector=df.col1)

Output: 1
```
### Median
#### Algorithm for calculating the median value
```python
from neulab.Algorithms import Median
d = {'col1': [0, 1, 2, 3]}

df = pd.DataFrame(data=d)
median = Median(vector=df.col1)

Output: 1.5
```
### Mode
#### Mode Calculation Algorithm
```python
from neulab.Algorithms import Mode

d = {'col1': [0, 1, 2, 3, 1]}
df = pd.DataFrame(data=d)

mode = Mode(vector=df.col1)

Output: 1
```
### Standart Deviation
#### Standard Deviation Algorithm
```python
from neulab.Algorithms import StdDeviation

d = {'col1': [8.02, 8.16, 3.97, 8.64, 0.84, 4.46, 0.81, 7.74, 8.78, 9.26, 20.46, 29.87, 10.38, 25.71]}
df = pd.DataFrame(data=d)
std = StdDeviation(df.col1)

Output: 8.767464705525615
```
### Is Symmetric
#### Detects if vector is symmetric or asymmetric.
```python
from neulab.Algorithms import IsSymmetric

d = {'col1': [8.02, 8.16, 3.97, 8.64, 0.84, 4.46, 0.81, 7.74, 8.78, 9.26, 20.46, 29.87, 10.38, 25.71]}
df = pd.DataFrame(data=d)
symmtr = IsSymmetric(vector=df.col1)

Output: True
```
### Euclid metric
#### Algorithm for calculating the distance using the Euclidean metric
```python
from neulab.Algorithms import EuclidMertic

d = {'col1': [0, 1, 2], 'col2': [2, 1, 0]}
df = pd.DataFrame(data=d)

euqld  = EuclidMertic(vector1=df.col1, vector2=df.col2) 

Output: 2.8284271247461903
```
### Manhattan metric
#### Algorithm for calculating the distance using the Manhattan metric
```python
from neulab.Algorithms import ManhattanMetric

d = {'col1': [0, 1, 2], 'col2': [2, 1, 0]}
df = pd.DataFrame(data=d)

mnht = ManhattanMetric(vector1=df.col1, vector2=df.col2) 

Output: 4.0
```
### Max Metric
#### Algorithm for calculating the distance using the Max metric
```python
from neulab.Algorithms import MaxMetric

d = {'col1': [0, 1, 2], 'col2': [2, 1, 0]}
df = pd.DataFrame(data=d)

mx = MaxMetric(vector1=df.col1, vector2=df.col2)

Output: 2
```
### Correlation Coefficient
#### Algorithm for calculating the correlation coefficient
```python
from neulab.Algorithms import CorrelationCoefficient

d = {'col1': [99, 89, 91, 91, 86, 97], 'col2': [58, 48, 54, 54, 44, 56]}
df = pd.DataFrame(data=d)

cc = CorrelationCoefficient(vector1=df.col1, vector2=df.col2)

Output: 0.906843948104356
```

## Restore value methods
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
from neulab.RestoreValue import MetricRestore

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
from neulab.RestoreValue import CorrCoefRestore

d = {'G': [99, 89, 91, 91, 86, 97, np.NaN], 'T': [56, 58, 64, 51, 56, 53, 51], 'B': [91, 89, 91, 91, 84, 86, 91], 'R': [160, 157, 165, 170, 157, 175, 165], 'W': [58, 48, 54, 54, 44, 56, 54]}
df = pd.DataFrame(data=d)

cc = CorrCoefRestore(df=df, row_start=0, row_end=9)

Output: 94.21
```

## Outliers Detection
### Simple Outlier Detection Algorithm
#### Algorithm detects and removes (if autorm is True) rows containing the outlier. Returns cleared dataframe.
```python
from neulab.OutlierDetection import SimpleOutDetect

d = {'col1': [1, 0, 342, 1, 1, 0, 1, 0, 1, 255, 1, 1, 1, 0, ]}
df = pd.DataFrame(data=d)

sd = SimpleOutDetect(dataframe=df, info=False, autorm=True)

Output: Detected outliers: {'col1': [342, 255]}

index	col1
0	   1
1	   0
3	   1
4	   1
5	   0 
6	   1
7	   0
8	   1
10	   1
11	   1
12	   1
13	   0
```

### Chauvenet Algorithm
#### Chauvenet Algorithm detects and removes (if autorm is True) rows containing the outlier. Returns cleared dataframe.
```python
from neulab.OutlierDetection import Chauvenet

d = {'col1': [8.02, 8.16, 3.97, 8.64, 0.84, 4.46, 0.81, 7.74, 8.78, 9.26, 20.46, 29.87, 10.38, 25.71], 'col2': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
df = pd.DataFrame(data=d)

chvn = Chauvenet(dataframe=df, info=True, autorm=True)

Output: Detected outliers: {'col1': [29.87, 25.71, 20.46, 0.84, 0.81, 3.97, 4.46, 10.38, 7.74, 9.26]}

	col1	col2
0	8.02	1
1	8.16	1
3	8.64	1
8	8.78	1
```
### Quratile algorithm
#### Quratile algorithm doest use standart deviation and average mean. Remove all outliers from the vector. Returns cleared dataframe is autorm is True.

```python
from neulab.OutlierDetection import Quratile

d = {'col1': [-6, 0, 1, 2, 4, 5, 5, 6, 7, 100], 'col2': [-1, 0, 1, 2, 0, 0, 1, 0, 50, 13]}
df = pd.DataFrame(data=d)

qurtl = Quratile(dataframe=df, info=True, autorm=True)

Output: Detected outliers: {'col1': [-6, 100], 'col2': [50]}

index col1	col2
1	   0	   0
2	   1	   1
3	   2	   2
4	   4	   0
5	   5	   0
6	   5	   1
7	   6	   0
```
### Metric algorithm
#### An outlier search algorithm using metrics. The metrics calculate the distance between features and then filter using the quantile algorithm. Returns cleared dataframe if autorm is True.
```python
from neulab.OutlierDetection import DistQuant

d = {'col1': [-6, 0, 1, 2, 4, 5, 5, 6, 7, 100], 'col2': [-1, 0, 1, 2, 0, 0, 1, 0, 50, 13]}
df = pd.DataFrame(data=d)

mdist = DistQuant(dataframe=df, metric='manhattan', filter='quantile', info=True, autorm=True)

Output: Distances: {0: 260.0, 1: 204.0, 2: 198.0, 3: 198.0, 4: 190.0, 5: 190.0, 6: 190.0, 7: 194.0, 8: 566.0, 9: 1014.0}

index col1	col2
1	   0	0
2	   1	1
3          2    2
4	   4	0
5	   5	0
6	   5	1
7	   6	0
```
