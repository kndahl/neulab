import sys
sys.path.append('.')
from neulib.RestoreValue import *
import pandas as pd

def test_MetricsEuclid():
    d = {'P1': [3, 5, 4, 5], 'P2': [4, 5, 3, 4], 'P3': [5, 5, 3, 3], 'P4': [3, 4, 2, 3], 'P5': [4, 3, 5, np.NaN]}
    df = pd.DataFrame(data=d)
    assert MetricRestore(df, row_start=0, row_end=8, metric='euclid') == 4.13

def test_MetricsMnhtn():
    d = {'P1': [3, 5, 4, 5], 'P2': [4, 5, 3, 4], 'P3': [5, 5, 3, 3], 'P4': [3, 4, 2, 3], 'P5': [4, 3, 5, np.NaN]}
    df = pd.DataFrame(data=d)
    assert MetricRestore(df, row_start=0, row_end=8, metric='manhattan') == 4.1

def test_MetricsMax():
    d = {'P1': [3, 5, 4, 5], 'P2': [4, 5, 3, 4], 'P3': [5, 5, 3, 3], 'P4': [3, 4, 2, 3], 'P5': [4, 3, 5, np.NaN]}
    df = pd.DataFrame(data=d)
    assert MetricRestore(df, row_start=0, row_end=8, metric='max') == 4.25

def test_CorCoef():
    d = {'G': [99, 89, 91, 91, 86, 97, np.NaN], 'T': [56, 58, 64, 51, 56, 53, 51], 'B': [91, 89, 91, 91, 84, 86, 91], 'R': [160, 157, 165, 170, 157, 175, 165], 'W': [58, 48, 54, 54, 44, 56, 54]}
    df = pd.DataFrame(data=d)
    assert CorrCoefRestore(df=df, row_start=0, row_end=8) == 94.21