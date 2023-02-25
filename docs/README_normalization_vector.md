# Normalize data using min max normalizer
### Function returns the normalized vector
```python
from neulab.Vector.normalization import min_max_normalizer

v1 = np.array([1, 2, 3, 4, 5])
v2 = np.array([10, 20, 30, 40, 50])
v3 = np.array([0, 5, 10, 15, 20])

vector = np.vstack((v1, v2, v3))

normalized_vec = min_max_normalizer(vector)

Output: 
[array([[0.02, 0.04, 0.06, 0.08, 0.1 ],
        [0.2 , 0.4 , 0.6 , 0.8 , 1.  ],
        [0.  , 0.1 , 0.2 , 0.3 , 0.4 ]])]
```
# Normalize data using mean normalizer
### Function returns the normalized vector
```python
from neulab.Vector.normalization import mean_normalizer

v1 = np.array([1, 2, 3, 4, 5])
v2 = np.array([10, 20, 30, 40, 50])
v3 = np.array([0, 5, 10, 15, 20])

vector = np.vstack((v1, v2, v3))

normalized_vec = mean_normalizer(vector)

Output: 
[array([[-0.90956084, -0.84134378, -0.77312672, -0.70490965, -0.63669259],
        [-0.29560727,  0.38656336,  1.06873399,  1.75090463,  2.43307526],
        [-0.97777791, -0.63669259, -0.29560727,  0.04547804,  0.38656336]])]
```