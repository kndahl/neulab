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