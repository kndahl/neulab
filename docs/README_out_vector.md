# Z-score outliers detection
### Finds outliers in n vectors using z-score algorithm.
-----
Please note that the z-score algorithm may not perform well on small vectors. Recommended minimum vector length is 11.
```python
import random 
from neulab.Vector.outliers import zscore_outliers

v1 = [1, 2, 54, 3, 0, -1, 2, 8, -2, 5, 2, 1]
v2 = [1, 2, 1, 3, 89, 0, 1, 0, 5, 4, 2]
v3 = [random.randint(0, 10) for i in range(50)]

cleared_vector, outliers = zscore_outliers(v1, v2, v3)

print("Cleared vector:", cleared_vector)
print("Outliers:", outliers)
Output:
Cleared vector: [
   [1, 2, 3, 0, -1, 2, 8, -2, 5, 2, 1], 
   [1, 2, 1, 3, 0, 1, 0, 5, 4, 2], 
   [0, 4, 6, 3, 7, 1, 9, 3, 4, 1, 9, 4, 9, 3, 8, 7, 7, 6, 4, 0, 5, 4, 7, 6, 7, 7, 3, 7, 6, 5, 2, 1, 7, 9, 3, 7, 1, 1, 3, 3, 10, 6, 2, 5, 1, 1, 2, 3, 5, 0]]
Outliers: [54, 89]
```
# Chauvenet outliers detection
### Finds outliers in n vectors using Chauvenet algorithm.
```python
import random 
from neulab.Vector.outliers import chauvenet_outliers

v1 = [1, 2, 54, 3, 5, 2, 1]
v2 = [1, 2, 1, 3, 89, 2]

cleared_vector, outliers = chauvenet_outliers(v1, v2)

print("Cleared vector:", cleared_vector)
print("Outliers:", outliers)
Output:
Cleared vectors: [[1, 2, 3, 5, 2, 1], [1, 2, 1, 3, 89, 2]]
Outliers: [54 89]
```
# Quartile outliers detection
### Finds outliers in n vectors using Quartile algorithm.
```python
import random 
from neulab.Vector.outliers import quartile_outliers

v1 = [1, 2, 54, 3, 5, 2, 1]
v2 = [1, 2, 1, 3, 89, 2]

cleared_vector, outliers = quartile_outliers(v1, v2)

print("Cleared vector:", cleared_vector)
print("Outliers:", outliers)
Output:
Cleared vectors: [[1, 2, 3, 5, 2, 1], [1, 2, 1, 3, 2]]
Outliers: [54, 89]
```
# Dixon Q-test outliers detection
### Finds outliers in n vectors using Dixon Q-test algorithm.
```python
import random 
from neulab.Vector.outliers import dixon_test_outliers

v1 = [1, 2, 54, 3, 5, 2, 1]
v2 = [1, 2, 1, 3, 89, 2]

cleared_vector, outliers = dixon_test_outliers(v1, v2)

print("Cleared vector:", cleared_vector)
print("Outliers:", outliers)
Output:
Cleared vectors: [[1, 2, 3, 5, 2, 1], [1, 2, 1, 3, 2]]
Outliers: [54, 89]
```