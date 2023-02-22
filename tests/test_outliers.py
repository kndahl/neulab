from neulab.outliers import zscore_outliers
from neulab.outliers import chauvenet_outliers
from neulab.outliers import quartile_outliers
from neulab.outliers import dixon_test_outliers
import numpy as np
import random

def test_zscore_outlier():
    v1 = [1, 2, 54, 3, 0, -1, 2, 8, -2, 5, 2, 1]
    v2 = [1, 2, 1, 3, 89, 0, 1, 0, 5, 4, 2]
    cleared_vector, outliers = zscore_outliers(v1, v2)
    assert cleared_vector == [[1, 2, 3, 0, -1, 2, 8, -2, 5, 2, 1], [1, 2, 1, 3, 0, 1, 0, 5, 4, 2]]
    assert outliers == [54, 89]

def test_chauvenet_outliers():
    v1 = [1, 2, 54, 3, 0, -1, 2, 8, -2, 5, 2, 1]
    v2 = [1, 2, 1, 3, 89, 0, 1, 0, 5, 4, 2]
    cleared_vector, outliers = chauvenet_outliers(v1, v2)
    assert cleared_vector == [[1, 2, 3, 0, -1, 2, 8, -2, 5, 2, 1], [1, 2, 1, 3, 89, 0, 1, 0, 5, 4, 2]]
    assert np.allclose(outliers, [54, 89])

def test_quartile_outliers():
    v1 = [1, 2, 54, 3, 5, 2, 1]
    v2 = [1, 2, 1, 3, 89, 2]
    cleared_vector, outliers = quartile_outliers(v1, v2)
    assert cleared_vector == [[1, 2, 3, 5, 2, 1], [1, 2, 1, 3, 2]]
    assert np.allclose(outliers, [54, 89])

def test_dixon_test_outliers():
    v1 = [1, 2, 54, 3, 5, 2, 1]
    v2 = [1, 2, 1, 3, 89, 2]
    cleared_vector, outliers = dixon_test_outliers(v1, v2)
    assert cleared_vector == [[1, 2, 3, 5, 2, 1], [1, 2, 1, 3, 2]]
    assert np.allclose(outliers, [54, 89])