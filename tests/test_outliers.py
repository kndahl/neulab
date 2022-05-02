import sys
sys.path.append('.')
from neulib.OutlierDetection import SimpleOutDetect
import pandas as pd
import numpy as np

def test_simple():

    d = {'col1': [1, 0, 1, 1, 0, 0, 1, 342, 1, 1, 0, 1, 0, 1, 255, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0 , 0 , 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 55, 0, 1], 'col2': [1, 0, 1, 1, 0, 0, 1, 0, 2, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0 , 0 , 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1]}
    df = pd.DataFrame(data=d)
    df_ethalon = SimpleOutDetect(dataframe=df, info=False, autorm=True)
    pd.testing.assert_frame_equal(SimpleOutDetect(dataframe=df, info=False, autorm=True), df_ethalon)