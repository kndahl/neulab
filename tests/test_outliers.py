from neulab.Vector.outliers import zscore_outliers
from neulab.Vector.outliers import chauvenet_outliers
from neulab.Vector.outliers import quartile_outliers
from neulab.Vector.outliers import dixon_test_outliers
from neulab.Dataframe.outliers import zscore
from neulab.Dataframe.outliers import chauvenet
import pandas as pd
import numpy as np

def test_zscore_outlier():
    v1 = [1, 2, 54, 3, 0, -1, 2, 8, -2, 5, 2, 1]
    v2 = [1, 2, 1, 3, 89, 0, 1, 0, 5, 4, 2]
    cleared_vector, outliers = zscore_outliers(v1, v2)
    assert cleared_vector == [[1, 2, 3, 0, -1, 2, -2, 5, 2, 1], [1, 2, 1, 3, 0, 1, 0, 5, 4, 2]]
    assert outliers == [54, 8, 89]

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

def test_zscore_frame():

    # Create a sample DataFrame
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4, 100, 6, 7, 8, 9, 10, 100],
        'col2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, -100],
        'col3': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10]
    })

    # Test the function without plotting
    result = zscore(df, plot=False)

    assert result == {'col1': [100, 100], 'col2': [-100], 'col3': [10]}

def test_chauvenet_frame():

    # Create a sample DataFrame
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4, 100, 6, 7, 8, 9, 10, 100],
        'col2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, -100],
        'col3': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10]
    })

    # Test the function without plotting
    result = chauvenet(df, plot=False)

    assert result == {'col1': [100, 100], 'col2': [-100], 'col3': [10]}