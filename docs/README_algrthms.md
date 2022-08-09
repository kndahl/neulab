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
from neulab.Algorithms import EuclidMetric

d = {'col1': [0, 1, 2], 'col2': [2, 1, 0]}
df = pd.DataFrame(data=d)

euqld  = EuclidMetric(vector1=df.col1, vector2=df.col2) 

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