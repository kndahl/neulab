import sys
sys.path.append('.')
from neulab.OutlierDetection import SimpleOutDetect, Chauvenet, Quratile, DistQuant, DixonTest
import pandas as pd
import numpy as np

def test_simple():
    d = {'col1': [1, 0, 1, 1, 0, 0, 1, 342, 1, 1, 0, 1, 0, 1, 255, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0 , 0 , 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 55, 0, 1], 'col2': [1, 0, 1, 1, 0, 0, 1, 0, 2, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0 , 0 , 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1]}
    df = pd.DataFrame(data=d)
    df_ethalon = SimpleOutDetect(dataframe=df, info=False, autorm=True)
    pd.testing.assert_frame_equal(SimpleOutDetect(dataframe=df, info=False, autorm=True), df_ethalon)

def test_chauvenet():
    d = {'col1': [8.02, 8.16, 3.97, 8.64, 0.84, 4.46, 0.81, 7.74, 8.78, 9.26, 20.46, 29.87, 10.38, 25.71], 'col2': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    df = pd.DataFrame(data=d)
    df_ethalon = pd.DataFrame(data={'col1': [8.02, 8.16, 8.64, 8.78], 'col2': [1, 1, 1, 1]}, index=[0, 1, 3, 8])
    pd.testing.assert_frame_equal(Chauvenet(dataframe=df, info=False, autorm=True), df_ethalon)

def test_quartile():
    d = {'col1': [-6, 0, 1, 2, 4, 5, 5, 6, 7, 100], 'col2': [-1, 0, 1, 2, 0, 0, 1, 0, 50, 13]}
    df = pd.DataFrame(data=d)
    df_ethalon = pd.DataFrame(data={'col1': [0, 1, 2, 4, 5, 5, 6], 'col2': [0, 1, 2, 0, 0, 1, 0]}, index=[1, 2, 3, 4, 5, 6, 7])
    pd.testing.assert_frame_equal(Quratile(dataframe=df, info=True, autorm=True), df_ethalon)

def test_metric():
    d = {'col1': [-6, 0, 1, 2, 4, 5, 5, 6, 7, 100], 'col2': [-1, 0, 1, 2, 0, 0, 1, 0, 50, 13]}
    df = pd.DataFrame(data=d)
    df_ethalon = pd.DataFrame(data={'col1': [0, 1, 2, 4, 5, 5, 6], 'col2': [0, 1, 2, 0, 0, 1, 0]}, index=[1, 2, 3, 4, 5, 6, 7])
    pd.testing.assert_frame_equal(DistQuant(dataframe=df, metric='manhattan', filter='quantile', info=True, autorm=True), df_ethalon)

def test_qtest():
    d = {'col1': [2131, 180, 188, 177, 181, 185, 189], 'col2': [0, 0, 0, 0, 1, 13, 1]}
    df = pd.DataFrame(data=d)
    df_ethalon = pd.DataFrame(data={'col1': [180, 188, 177, 181, 189], 'col2': [0, 0, 0, 1, 1]}, index=[1, 2, 3, 4, 6])
    pd.testing.assert_frame_equal(DixonTest(dataframe=df, q=95, info=True, autorm=True), df_ethalon)