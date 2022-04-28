import sys
sys.path.append('.')
from neulib.Normalization import *
import pandas as pd
import numpy as np

def test_InterNorm():
    d = {'col1': [1, 0, 5, 2, 2]}
    df = pd.DataFrame(data=d)
    np.testing.assert_array_equal(InterNormalization(df.col1), np.array([0.2, 0. , 1. , 0.4, 0.4]))

def test_MeanNorm():
    d = {'col1': [1, 0, 5, 2, 2]}
    df = pd.DataFrame(data=d)
    np.testing.assert_almost_equal(MeanNormalization(df.col1), np.array([-0.5976143 , -1.19522861,  1.79284291,  0.        ,  0.        ]))