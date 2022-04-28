import sys
sys.path.append('.')
from neulib.Algorithms import *
import pandas as pd

def test_Mean():
    d = {'col1': [0, 1, 2]}
    df = pd.DataFrame(data=d)
    assert Mean(vector=df.col1) == 1.0

def test_Median():
    d = {'col1': [0, 1, 2, 3]}
    df = pd.DataFrame(data=d)
    assert Median(vector=df.col1) == 1.5

def test_Mode():
    d = {'col1': [0, 1, 2, 3, 1]}
    df = pd.DataFrame(data=d)
    assert Mode(vector=df.col1) == 1

def test_Euclid():
    d = {'col1': [0, 1, 2], 'col2': [2, 1, 0]}
    df = pd.DataFrame(data=d)
    assert EuclidMertic(vector1=df.col1, vector2=df.col2) == 2.8284271247461903

def test_Manhattan():
    d = {'col1': [0, 1, 2], 'col2': [2, 1, 0]}
    df = pd.DataFrame(data=d)
    assert ManhattanMetric(vector1=df.col1, vector2=df.col2) == 4.0

def test_MaxMetric():
    d = {'col1': [0, 1, 2], 'col2': [2, 1, 0]}
    df = pd.DataFrame(data=d)
    assert MaxMetric(vector1=df.col1, vector2=df.col2) == 2

def test_CorCoef():
    d = {'col1': [99, 89, 91, 91, 86, 97], 'col2': [58, 48, 54, 54, 44, 56]}
    df = pd.DataFrame(data=d)
    assert CorrelationCoefficient(vector1=df.col1, vector2=df.col2) == 0.906843948104356