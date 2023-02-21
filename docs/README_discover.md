# Euclidean distance
### Function returns Euclidean distance between all pairs of vectors.
```python
from neulab.discover import euclidean_distance

v1 = [0, 1, 2]
v2 = [2, 1, 0]
v3 = [1, 0, 1]

dist = euclidean_distance(v1, v2, v3)

Output: [2.8284271247461903, 1.7320508075688772, 1.7320508075688772]
# This can be interpreted as:
dist between v1 and v2: 2.828427124746190
dist between v1 and v3: 1.7320508075688772
dist between v2 and v3: 1.7320508075688772
```


# Manhattan distance
### Function returns Manhattan distance between all pairs of vectors.
```python
from neulab.discover import manhattan_distance

v1 = [0, 1, 2]
v2 = [2, 1, 0]
v3 = [1, 0, 1]

dist = manhattan_distance(v1, v2, v3)

Output: [4, 3, 3]
# This can be interpreted as:
dist between v1 and v2: 4
dist between v1 and v3: 3
dist between v2 and v3: 3
```
# Max distance
### Function returns Max distance between all pairs of vectors.
```python
from neulab.discover import max_distance

v1 = [0, 1, 2]
v2 = [2, 1, 0]
v3 = [1, 0, 1]
dist = max_distance(v1, v2, v3)

Output: [2, 1, 1]
# This can be interpreted as:
dist between v1 and v2: 2
dist between v1 and v3: 1
dist between v2 and v3: 1
```
# Correlation coefficient
### Function returns correlation coefficient between all pairs.
```python
from neulab.discover import correlation_coefficient

v1 = [1, 0, 2]
v2 = [1, 1, 2]
v3 = [1, 0, 2]

corr_coef = correlation_coefficient(v1, v2, v3)

Output: [0.8660254037844387, 1.0, 0.8660254037844387]
# This can be interpreted as:
corr_coef between v1 and v2: 0.8660254037844387
corr_coef between v1 and v3: 1
corr_coef between v2 and v3: 0.8660254037844387
```
# Standard deviation
### Function returns standard deviation
```python
from neulab.discover import std_deviation

v1 = [1, 2, 3, 4, 5, 6]
v2 = [2, 13, 4, 0, 6, -7]
v3 = [0, -1, 5]

std = std_deviation(v1, v2)

Output: [1.8708286933869707, 6.6332495807108, 3.2145502536643185]
# This can be interpreted as:
std for v1: 1.8708286933869707
std for v2: 6.6332495807108
std for v3: 3.2145502536643185
```
