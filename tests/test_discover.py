import sys
sys.path.append('.')
from neulab.Vector.discover import euclidean_distance
from neulab.Vector.discover import manhattan_distance
from neulab.Vector.discover import max_distance
from neulab.Vector.discover import correlation_coefficient
from neulab.Vector.discover import std_deviation
from neulab.Dataframe.discover import get_categorical_columns
import pandas as pd
import numpy as np
import pytest
import random

def test_euclidean_same_points():
    vector1 = [1, 2, 3]
    vector2 = [1, 2, 3]
    assert euclidean_distance(vector1, vector2) == [0]

def test_euclidean_different_points():
    vector1 = [1, 2, 3]
    vector2 = [4, 5, 6]
    assert euclidean_distance(vector1, vector2) == [np.sqrt(27)]

def test_euclidean_invalid_input():
    vector1 = [1, 2, 3]
    vector2 = [1, 2]
    with pytest.raises(ValueError):
        euclidean_distance(vector1, vector2)

def test_manhattan_distance_same_points():
    vector1 = [1, 2, 3]
    vector2 = [1, 2, 3]
    print(manhattan_distance(vector1, vector2))
    assert manhattan_distance(vector1, vector2) == [0]

def test_manhattan_distance_different_points():
    vector1 = [1, 2, 3]
    vector2 = [4, 5, 6]
    print(manhattan_distance(vector1, vector2))
    assert manhattan_distance(vector1, vector2) == [9]

def test_manhattan_distance_invalid_input():
    vector1 = [1, 2, 3]
    vector2 = [1, 2]
    with pytest.raises(ValueError):
        manhattan_distance(vector1, vector2)

def test_max_distance_same_points():
    vector1 = [1, 2, 3]
    vector2 = [1, 2, 3]
    assert max_distance(vector1, vector2) == [0]

def test_max_distance_different_points():
    vector1 = [1, 2, 3]
    vector2 = [4, 5, 6]
    assert max_distance(vector1, vector2) == [3]

def test_max_distance_invalid_input():
    vector1 = [1, 2, 3]
    vector2 = [1, 2]
    with pytest.raises(ValueError):
        max_distance(vector1, vector2)

def test_correlation_coefficient_identical_vectors():
    v1 = np.array([1, 2, 3, 4])
    v2 = np.array([1, 2, 3, 4])
    assert round(correlation_coefficient(v1, v2)[0], 6) == 1.0

def test_correlation_coefficient_orthogonal_vectors():
    v1 = np.array([1, 0, 0, 1])
    v2 = np.array([0, 1, 1, 0])
    assert np.isclose(correlation_coefficient(v1, v2)[0], -1.0)

def test_correlation_coefficient_high():
    v1 = np.array([1, 2, 3, 4])
    v2 = np.array([2, 4, 6, 8])
    assert np.isclose(correlation_coefficient(v1, v2)[0], 1.0)

def test_correlation_coefficient_low():
    v1 = np.array([1, 2, 3, 4])
    v2 = np.array([4, 3, 2, 1])
    assert np.isclose(correlation_coefficient(v1, v2)[0], -1.0)

def test_correlation_coefficient_random():
    v1 = np.random.rand(100)
    v2 = np.random.rand(100)
    assert -1 <= correlation_coefficient(v1, v2)[0] <= 1

def test_std():
    v = [8.02, 8.16, 3.97, 8.64, 0.84, 4.46, 0.81, 7.74, 8.78, 9.26, 20.46, 29.87, 10.38, 25.71]
    assert std_deviation(v) == 8.767464705525615

def test_get_categorical():
    df = pd.DataFrame({
    'col1': [random.randint(0, 99) for _ in range(50)],
    'col2': [random.randint(0, 100) for _ in range(50)],
    'col3': [random.randint(50, 90) for _ in range(50)],
    'col4': [random.randint(0, 1) for _ in range(50)]
    })

    output = get_categorical_columns(df)
    assert output == ['col4']