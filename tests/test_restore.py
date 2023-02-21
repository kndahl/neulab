import sys
sys.path.append('.')
from neulab.recover import replace_missing_with_mean
from neulab.recover import replace_missing_with_median
from neulab.recover import replace_missing_with_mode
from neulab.recover import replace_missing_with_corrcoef
from neulab.recover import replace_missing_with_distance
import numpy as np
import pytest

def test_replace_missing_with_mean_all_missing():
    vector = np.full((5,), np.nan)
    try:
        replace_missing_with_mean(vector)
    except ValueError as e:
        assert str(e) == "All values in vector are missing."

def test_replace_missing_with_mean_no_missing():
    vector = np.array([1, 2, 3, 4, 5])
    new_vector, mean_value = replace_missing_with_mean(vector)
    assert np.allclose(new_vector, vector)
    assert mean_value == 3.0

def test_replace_missing_with_mean_some_missing():
    vector = np.array([1, np.nan, 3, np.nan, 5])
    new_vector, mean_value = replace_missing_with_mean(vector)
    assert np.allclose(new_vector, np.array([1, 3, 3, 3, 5]))
    assert mean_value == 3.0

def test_replace_missing_with_median_all_missing():
    vector = np.array([np.nan, np.nan, np.nan])
    with pytest.raises(ValueError):
        replace_missing_with_median(vector)

def test_replace_missing_with_median_no_missing():
    vector = np.array([1, 2, 3, 4, 5])
    new_vector, median = replace_missing_with_median(vector)
    assert np.array_equal(new_vector, vector)
    assert median == 3

def test_replace_missing_with_median_some_missing():
    vector = np.array([1, 2, np.nan, 4, np.nan])
    new_vector, median = replace_missing_with_median(vector)
    expected_vector = np.array([1., 2., 2., 4., 2.])
    expected_median = 2.0
    assert np.array_equal(new_vector, expected_vector)
    assert median == expected_median

def test_replace_missing_with_mode_no_missing():
    vector = [1, 2, 3, 4, 5]
    new_vector, mode_value = replace_missing_with_mode(vector)
    assert np.allclose(new_vector, vector)
    assert mode_value == 1

def test_replace_missing_with_mode_some_missing():
    vector = [1, 2, np.nan, 4, np.nan, 4]
    new_vector, mode_value = replace_missing_with_mode(vector)
    expected = [1, 2, 4, 4, 4, 4]
    assert np.allclose(new_vector, expected)
    assert mode_value == 4

def test_replace_missing_with_mode_all_missing():
    vector = [np.nan, np.nan, np.nan]
    with pytest.raises(ValueError):
        replace_missing_with_mode(vector)

def test_replace_missing_with_corrcoef_all_missing():
    vector = np.full((5,), np.nan)
    try:
        replace_missing_with_corrcoef(vector)
    except ValueError as e:
        assert str(e) == "All values in vector are missing."

def test_replace_missing_with_corrcoef_no_missing():
    vector = np.array([1, 2, 3, 4, 5])
    new_vector, corr_coef = replace_missing_with_corrcoef(vector)
    assert np.allclose(new_vector, vector)
    assert np.isnan(corr_coef)

def test_replace_missing_with_corrcoef_some_missing():
    vector = np.array([1, 2, np.nan, 4, np.nan, 6, np.nan, np.nan])
    new_vector, corr_coef = replace_missing_with_corrcoef(vector)
    assert np.allclose(new_vector, np.array([1., 2., 3., 4., 5., 6., 7., 8.]))
    assert corr_coef == 1.0

def test_replace_missing_with_distance_no_missing():
    v1 = np.array([1, 2, 3, 4, 5])
    v2 = np.array([3, 2, 3, 1, 5])
    vector = np.vstack((v1, v2))
    new_vector, dist = replace_missing_with_distance(vector)
    assert np.allclose(new_vector, vector)
    assert np.isnan(dist)

def test_replace_missing_with_distance_all_missing():
    vector = np.full((5,), np.nan)
    try:
        replace_missing_with_distance(vector)
    except ValueError as e:
        assert str(e) == "All values in vectors are missing."

def test_replace_missing_with_distance_some_missing():
    v1 = [3, 5, 4, 5]
    v2 = [4, 5, 3, 4]
    v3 = [5, 5, 3, 3]
    v4 = [3, 4, 2, 3]
    v5 = [4, 3, 5, np.nan]
    vector = np.vstack((v1, v2, v3, v4, v5))
    new_vector, dist = replace_missing_with_distance(vector, metric='euclidean', how='vertical')
    assert np.allclose(new_vector, np.array([[3.  , 5.  , 4.  , 5.  ],
        [4.  , 5.  , 3.  , 4.  ],
        [5.  , 5.  , 3.  , 3.  ],
        [3.  , 4.  , 2.  , 3.  ],
        [4.  , 3.  , 5.  , 4.13]]))
    assert dist == 4.13

    v1 = [3, 5, 4, 5]
    v2 = [4, 5, 3, 4]
    v3 = [5, 5, 3, 3]
    v4 = [3, 4, 2, 3]
    v5 = [4, 3, 5, np.nan]
    vector = np.vstack((v1, v2, v3, v4, v5))
    new_vector, dist = replace_missing_with_distance(vector, metric='euclidean', how='horizontal')
    assert np.allclose(new_vector, np.array([[3.  , 5.  , 4.  , 5.  ],
        [4.  , 5.  , 3.  , 4.  ],
        [5.  , 5.  , 3.  , 3.  ],
        [3.  , 4.  , 2.  , 3.  ],
        [4.  , 3.  , 5.  , 3.12]]))
    assert dist == 3.12

    v1 = [3, 5, 4, 5]
    v2 = [4, 5, 3, 4]
    v3 = [5, 5, 3, 3]
    v4 = [3, 4, 2, 3]
    v5 = [4, 3, 5, np.nan]
    vector = np.vstack((v1, v2, v3, v4, v5))
    new_vector, dist = replace_missing_with_distance(vector, metric='manhattan', how='vertical')
    assert np.allclose(new_vector, np.array([[3. , 5. , 4. , 5. ],
        [4. , 5. , 3. , 4. ],
        [5. , 5. , 3. , 3. ],
        [3. , 4. , 2. , 3. ],
        [4. , 3. , 5. , 4.1]]))
    assert dist == 4.1

    v1 = [3, 5, 4, 5]
    v2 = [4, 5, 3, 4]
    v3 = [5, 5, 3, 3]
    v4 = [3, 4, 2, 3]
    v5 = [4, 3, 5, np.nan]
    vector = np.vstack((v1, v2, v3, v4, v5))
    new_vector, dist = replace_missing_with_distance(vector, metric='max', how='vertical')
    assert np.allclose(new_vector, np.array([[3.  , 5.  , 4.  , 5.  ],
        [4.  , 5.  , 3.  , 4.  ],
        [5.  , 5.  , 3.  , 3.  ],
        [3.  , 4.  , 2.  , 3.  ],
        [4.  , 3.  , 5.  , 4.25]]))
    assert dist == 4.25