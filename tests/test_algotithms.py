import sys
sys.path.append('.')
from neulab.Algorithms import replace_missing_with_mean
from neulab.Algorithms import replace_missing_with_median
from neulab.Algorithms import replace_missing_with_mode
from neulab.Algorithms import euclidean_distance
from neulab.Algorithms import manhattan_distance
from neulab.Algorithms import max_metric
from neulab.Algorithms import correlation_coefficient
from neulab.Algorithms import std_deviation
from neulab.Algorithms import is_symmetric
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

def test_euclidean_same_points():
    vector1 = [1, 2, 3]
    vector2 = [1, 2, 3]
    assert euclidean_distance(vector1, vector2) == 0

def test_euclidean_different_points():
    vector1 = [1, 2, 3]
    vector2 = [4, 5, 6]
    assert euclidean_distance(vector1, vector2) == np.sqrt(27)

def test_euclidean_invalid_input():
    vector1 = [1, 2, 3]
    vector2 = [1, 2]
    with pytest.raises(ValueError):
        euclidean_distance(vector1, vector2)

def test_manhattan_distance_same_points():
    vector1 = [1, 2, 3]
    vector2 = [1, 2, 3]
    print(manhattan_distance(vector1, vector2))
    assert manhattan_distance(vector1, vector2) == 0

def test_manhattan_distance_different_points():
    vector1 = [1, 2, 3]
    vector2 = [4, 5, 6]
    print(manhattan_distance(vector1, vector2))
    assert manhattan_distance(vector1, vector2) == 9

def test_manhattan_distance_invalid_input():
    vector1 = [1, 2, 3]
    vector2 = [1, 2]
    with pytest.raises(ValueError):
        manhattan_distance(vector1, vector2)

def test_max_metric_same_points():
    vector1 = [1, 2, 3]
    vector2 = [1, 2, 3]
    assert max_metric(vector1, vector2) == 0

def test_max_metric_different_points():
    vector1 = [1, 2, 3]
    vector2 = [4, 5, 6]
    assert max_metric(vector1, vector2) == 3

def test_max_metric_invalid_input():
    vector1 = [1, 2, 3]
    vector2 = [1, 2]
    with pytest.raises(ValueError):
        max_metric(vector1, vector2)

def test_correlation_coefficient_identical_vectors():
    v1 = np.array([1, 2, 3, 4])
    v2 = np.array([1, 2, 3, 4])
    assert round(correlation_coefficient(v1, v2), 6) == 1.0

def test_correlation_coefficient_orthogonal_vectors():
    v1 = np.array([1, 0, 0, 1])
    v2 = np.array([0, 1, 1, 0])
    assert np.isclose(correlation_coefficient(v1, v2), -1.0)

def test_correlation_coefficient_high():
    v1 = np.array([1, 2, 3, 4])
    v2 = np.array([2, 4, 6, 8])
    assert np.isclose(correlation_coefficient(v1, v2), 1.0)

def test_correlation_coefficient_low():
    v1 = np.array([1, 2, 3, 4])
    v2 = np.array([4, 3, 2, 1])
    assert np.isclose(correlation_coefficient(v1, v2), -1.0)

def test_correlation_coefficient_random():
    v1 = np.random.rand(100)
    v2 = np.random.rand(100)
    assert -1 <= correlation_coefficient(v1, v2) <= 1

def test_std():
    v = [8.02, 8.16, 3.97, 8.64, 0.84, 4.46, 0.81, 7.74, 8.78, 9.26, 20.46, 29.87, 10.38, 25.71]
    assert std_deviation(v) == 8.767464705525615

def test_symmetrci():
    v = [8.02, 8.16, 3.97, 8.64, 0.84, 4.46, 0.81, 7.74, 8.78, 9.26, 20.46, 29.87, 10.38, 25.71]
    assert is_symmetric(vector=v) == True