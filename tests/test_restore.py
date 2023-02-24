import sys
sys.path.append('.')
from neulab.Vector.recover import replace_missing_with_mean
from neulab.Vector.recover import replace_missing_with_median
from neulab.Vector.recover import replace_missing_with_mode
from neulab.Vector.recover import replace_missing_with_corrcoef
from neulab.Vector.recover import replace_missing_with_distance
from neulab.Dataframe.recover import simple_imputation
from neulab.Dataframe.recover import iterative_imputation
from neulab.Dataframe.recover import distance_imputation
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
import pandas as pd
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
    assert corr_coef[0] == 1.0

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
    new_vector, dist = replace_missing_with_distance(vector, metric='euclidean')
    assert np.allclose(new_vector, np.array(([[3.  , 5.  , 4.  , 5.  ],
        [4.  , 5.  , 3.  , 4.  ],
        [5.  , 5.  , 3.  , 3.  ],
        [3.  , 4.  , 2.  , 3.  ],
        [4.  , 3.  , 5.  , 3.11]])))
    assert dist == [3.11]

    v1 = [3, np.nan, 4, 5]
    v2 = [4, 5, 3, 4]
    v3 = [5, 5, 3, 3]
    v4 = [3, 4, 2, 3]
    v5 = [4, np.nan, 5, np.nan]
    vector = np.vstack((v1, v2, v3, v4, v5))
    new_vector, dist = replace_missing_with_distance(vector, metric='euclidean')
    assert np.allclose(new_vector, np.array([[3.  , 2.52, 4.  , 5.  ],
        [4.  , 5.  , 3.  , 4.  ],
        [5.  , 5.  , 3.  , 3.  ],
        [3.  , 4.  , 2.  , 3.  ],
        [4.  , 2.61, 5.  , 2.61]]))
    assert dist == [2.52, 2.61]

    v1 = [3, 5, 4, 5]
    v2 = [4, 5, 3, 4]
    v3 = [5, 5, 3, 3]
    v4 = [3, 4, 2, 3]
    v5 = [4, 3, 5, np.nan]
    new_vector, dist = replace_missing_with_distance(v1, v2, v3, v4, v5, metric='manhattan')
    assert np.allclose(new_vector, np.array([[3.  , 5.  , 4.  , 5.  ],
        [4.  , 5.  , 3.  , 4.  ],
        [5.  , 5.  , 3.  , 3.  ],
        [3.  , 4.  , 2.  , 3.  ],
        [4.  , 3.  , 5.  , 3.04]]))
    assert dist == [3.04]

    v1 = [3, 5, np.nan, 5]
    v2 = [4, 5, 3, 4]
    v3 = [5, 5, 3, 3]
    v4 = [3, np.nan, 2, 3]
    v5 = [4, 3, 5, np.nan]

    new_vector, dist = replace_missing_with_distance(v1, v2, v3, v4, v5, metric='max')
    assert np.allclose(new_vector, np.array([[3.  , 5.  , 3.16, 5.  ],
        [4.  , 5.  , 3.  , 4.  ],
        [5.  , 5.  , 3.  , 3.  ],
        [3.  , 2.12, 2.  , 3.  ],
        [4.  , 3.  , 5.  , 3.27]]))
    assert dist == [3.16, 2.12, 3.27]

def test_simple_imputation_mean():
    # Create test dataframe with NaN values
    data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [6, 7, 8, np.nan, 10]
    })

    # Define expected output after imputation with mean
    expected_output = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [6, 7, 8, 7.75, 10]
    })

    # Call function to impute NaN values with mean
    output = simple_imputation(data, method='mean')

    expected_output = expected_output.astype(output.dtypes)

    # Assert that output is equal to expected output
    assert_frame_equal(output, expected_output)

def test_simple_imputation_median():
    # Create test dataframe with NaN values
    data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [6, 7, 8, np.nan, 10]
    })

    # Define expected output after imputation with median
    expected_output = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [6, 7, 8, 7.5, 10]
    })

    # Call function to impute NaN values with median
    output = simple_imputation(data, method='median')

    expected_output = expected_output.astype(output.dtypes)

    # Assert that output is equal to expected output
    assert_frame_equal(output, expected_output)

def test_simple_imputation_mode():
    # Create test dataframe with NaN values
    data = pd.DataFrame({
        'A': [1, 2, np.nan, np.nan, 5],
        'B': [6, 7, 7, np.nan, 10]
    })

    # Define expected output after imputation with mode
    expected_output = pd.DataFrame({
        'A': [1, 2, 1, 1, 5],
        'B': [6, 7, 7, 7, 10]
    })

    # Call function to impute NaN values with mode
    output = simple_imputation(data, method='mode')

    expected_output = expected_output.astype(output.dtypes)

    # Assert that output is equal to expected output
    assert_frame_equal(output, expected_output)

def test_simple_imputation_invalid_method():
    # Create test dataframe with NaN values
    data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [6, 7, 8, np.nan, 10]
    })

    # Call function with invalid method
    try:
        simple_imputation(data, method='invalid')
    except ValueError as e:
        # Assert that ValueError is raised with correct error message
        assert str(e) == "Invalid imputation method: invalid."
    else:
        # If no ValueError is raised, the test fails
        assert False

def test_distance_imputation_euclidean():
    # create a sample dataframe with missing values
    df = pd.DataFrame({'A': [1, 2, np.nan, 4], 'B': [5, np.nan, 7, 8]})
    # expected output after imputation
    expected_output = pd.DataFrame({'A': [1.0, 2.0, 1.0, 4.0], 'B': [5.0, 5.0, 7.0, 8.0]})
    # perform imputation
    output = distance_imputation(df, metric='euclidean')
    expected_output = expected_output.astype(output.dtypes)
    # compare actual output with expected output
    assert_frame_equal(output, expected_output)

def test_distance_imputation_manhattan():
    # create a sample dataframe with missing values
    df = pd.DataFrame({'A': [1, 2, np.nan, 4], 'B': [5, np.nan, 7, 8]})
    # expected output after imputation
    expected_output = pd.DataFrame({'A': [1.0, 2.0, 1.0, 4.0], 'B': [5.0, 5.0, 7.0, 8.0]})
    # perform imputation
    output = distance_imputation(df, metric='manhattan')
    expected_output = expected_output.astype(output.dtypes)
    # compare actual output with expected output
    assert_frame_equal(output, expected_output)

def test_distance_imputation_max():
    # create a sample dataframe with missing values
    df = pd.DataFrame({'A': [1, 2, np.nan, 4], 'B': [5, np.nan, 7, 8]})
    # expected output after imputation
    expected_output = pd.DataFrame({'A': [1.0, 2.0, 1.0, 4.0], 'B': [5.0, 5.0, 7.0, 8.0]})
    # perform imputation
    output = distance_imputation(df, metric='max')
    expected_output = expected_output.astype(output.dtypes)
    # compare actual output with expected output
    assert_frame_equal(output, expected_output)

def test_distance_imputation_invalid_metric():
    # create a sample dataframe with missing values
    df = pd.DataFrame({'A': [1, 2, np.nan, 4], 'B': [5, np.nan, 7, 8]})
    # check that an error is raised when an invalid metric is passed
    with pytest.raises(ValueError):
        distance_imputation(df, metric='invalid_metric')