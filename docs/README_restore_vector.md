# Restore data using mean value
### Function returns the new vector and the mean value.
```python
from neulab.Vector.recover import replace_missing_with_mean

v1 = [1, np.nan, 3, np.nan, 5, 5]
vector, mean_value = replace_missing_with_mean(v1)

Output: (array([1. , 3.5, 3. , 3.5, 5. , 5. ]), 3.5)
```
# Restore data using median value
### Function returns the new vector and the median value.
```python
from neulab.Vector.recover import replace_missing_with_median

v1 = [1, np.nan, 3, np.nan, 5, 5]
vector, median_value = replace_missing_with_median(v1)

Output: (array([1., 4., 3., 4., 5., 5.]), 4.0)
```
# Restore data using mode value
### Function returns the new vector and the mode value.
```python
from neulab.Vector.recover import replace_missing_with_mode

v1 = [1, np.nan, 3, np.nan, 5, 5]
vector, mode_value = replace_missing_with_mode(v1)

Output: (array([1., 5., 3., 5., 5., 5.]), 5.0)
```
# Restore data using correlation coefficient value
### Function returns the new vector and the correlation coefficient value.
```python
from neulab.Vector.recover import replace_missing_with_corrcoef

v1 = np.array([1, 2, np.nan, 4, np.nan, 6, np.nan, np.nan])
new_vector, corrvalue = replace_missing_with_corrcoef(v1)

Output: (array([1., 2., 3., 4., 5., 6., 7., 8.]), 1.0)
```
# Restore data using distance value
### Function returns the new vector and the distance value.
```python
from neulab.Vector.recover import replace_missing_with_distance

v1 = [3, 5, 4, 5]
v2 = [4, 5, 3, 4]
v3 = [5, 5, 3, 3]
v4 = [3, 4, 2, 3]
v5 = [4, 3, 5, np.nan]

new_vector, dist = replace_missing_with_distance(v1, v2, v3, v4, v5, metric='euclidean')

Output: 
(array([[3.  , 5.  , 4.  , 5.  ],
        [4.  , 5.  , 3.  , 4.  ],
        [5.  , 5.  , 3.  , 3.  ],
        [3.  , 4.  , 2.  , 3.  ],
        [4.  , 3.  , 5.  , 3.11]]),
 [3.11])
```