import sys
sys.path.append('.')
from neulab.Vector.normalization import min_max_normalizer
from neulab.Vector.normalization import mean_normalizer
from neulab.Dataframe.normalization import min_max_normalize
from neulab.Dataframe.normalization import z_score_normalize
from neulab.Dataframe.normalization import log_transform
from neulab.Dataframe.normalization import power_transform
import pandas as pd
import numpy as np

def test_min_max_normalizer():
    v1 = np.array([1, 2, 3, 4, 5])
    v2 = np.array([10, 20, 30, 40, 50])
    v3 = np.array([0, 5, 10, 15, 20])
    vector = np.vstack((v1, v2, v3))
    normalized_vec = min_max_normalizer(vector)
    assert np.allclose(normalized_vec, ([[0.02, 0.04, 0.06, 0.08, 0.1 ],
        [0.2 , 0.4 , 0.6 , 0.8 , 1.  ],
        [0.  , 0.1 , 0.2 , 0.3 , 0.4 ]]))

def test_mean_normalizer():
    v1 = np.array([1, 2, 3, 4, 5])
    v2 = np.array([10, 20, 30, 40, 50])
    v3 = np.array([0, 5, 10, 15, 20])
    vector = np.vstack((v1, v2, v3))
    normalized_vec = mean_normalizer(vector)
    assert np.allclose(normalized_vec,([[-0.90956084, -0.84134378, -0.77312672, -0.70490965, -0.63669259],
        [-0.29560727,  0.38656336,  1.06873399,  1.75090463,  2.43307526],
        [-0.97777791, -0.63669259, -0.29560727,  0.04547804,  0.38656336]]))
    
def test_min_max_normalize_frame():
    df = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [10, 20, np.nan, 40], 'col3': [100, 200, 300, np.nan]})
    df_normalized = min_max_normalize(df, cols_to_normalize=df.columns)
    expct_data = {'col1': {0: 0.0, 1: 0.3333333333333333, 2: 0.6666666666666666, 3: 1.0},
                'col2': {0: 0.0, 1: 0.3333333333333333, 2: np.nan, 3: 1.0},
                'col3': {0: 0.0, 1: 0.5, 2: 1.0, 3: np.nan}}
    df_expected = pd.DataFrame(data=expct_data)
    assert df_normalized.equals(df_expected)

def test_zscore_normalize_frame():
    df = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [10, 20, np.nan, 40], 'col3': [100, 200, 300, np.nan]})
    df_normalized = z_score_normalize(df, cols_to_normalize=df.columns)
    expct_data = {'col1': {0: -1.161895003862225, 1: -0.3872983346207417, 2: 0.3872983346207417, 3: 1.161895003862225},
                    'col2': {0: -0.8728715609439694, 1: -0.2182178902359923, 2: np.nan, 3: 1.091089451179962},
                    'col3': {0: -1.0, 1: 0.0, 2: 1.0, 3: np.nan}}
    df_expected = pd.DataFrame(data=expct_data)
    assert df_normalized.equals(df_expected)

def test_log_transform():
    df = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [10, 20, np.nan, 40], 'col3': [100, 200, 300, np.nan]})
    df_normalized = log_transform(df, cols_to_transform=df.columns)
    expct_data = {'col1': {0: 0.0, 1: 0.6931471805599453, 2: 1.0986122886681098, 3: 1.3862943611198906},
                    'col2': {0: 2.302585092994046, 1: 2.995732273553991, 2: np.nan, 3: 3.6888794541139363},
                    'col3': {0: 4.605170185988092, 1: 5.298317366548036, 2: 5.703782474656201, 3: np.nan}}
    df_expected = pd.DataFrame(data=expct_data)
    assert df_normalized.equals(df_expected)

def test_power_transdorm():
    df = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [10, 20, np.nan, 40], 'col3': [100, 200, 300, np.nan]})
    df_normalized = power_transform(df, cols_to_transform=df.columns, power=2)
    expct_data = {'col1': {0: 1, 1: 4, 2: 9, 3: 16},
                    'col2': {0: 100.0, 1: 400.0, 2: np.nan, 3: 1600.0},
                    'col3': {0: 10000.0, 1: 40000.0, 2: 90000.0, 3: np.nan}}
    df_expected = pd.DataFrame(data=expct_data)
    assert df_normalized.equals(df_expected)