from neulab.Vector.outliers import zscore_outliers
from neulab.Vector.outliers import chauvenet_outliers
from neulab.Vector.outliers import quartile_outliers
from neulab.Vector.outliers import dixon_test_outliers
from neulab.Dataframe.outliers import zscore
from neulab.Dataframe.outliers import chauvenet
import pandas as pd
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

def test_zscore_frame():

    c1 = [random.randint(0, 10) for _ in range(50)]
    c2 = [random.randint(10, 100) for _ in range(50)]
    c3 = [1 for _ in range(50)]

    # Append the values to the lists separately
    c1.append(100)
    c2.append(-100)
    c3.append(10)

    # Create the DataFrame with the lists
    df = pd.DataFrame({
        'col1': c1,
        'col2': c2,
        'col3': c3
    })

    # Test the function without plotting
    result = zscore(df, plot=False)

    assert result == {'col1': [100], 'col2': [-100], 'col3': [10]}

def test_chauvenet_frame():

    c1 = [random.randint(0, 10) for _ in range(50)]
    c2 = [random.randint(10, 100) for _ in range(50)]
    c3 = [1 for _ in range(50)]

    # Append the values to the lists separately
    c1.append(100)
    c2.append(-100)
    c3.append(10)

    # Create the DataFrame with the lists
    df = pd.DataFrame({
        'col1': c1,
        'col2': c2,
        'col3': c3
    })

    # Test the function without plotting
    result = chauvenet(df, plot=False)

    assert result == {'col1': [100], 'col2': [-100], 'col3': [10]}