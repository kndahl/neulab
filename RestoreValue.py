from calendar import c
import numpy as np
from Algorithms import *

def MeanRestore(column):
    '''Replace missing value with mean. Returns value to restore.'''
    return Mean(column=column)

def MedianRestore(column):
    '''Replace missing value with median. Returns value to restore.'''
    return Median(column=column)

def ModeRestore(column):
    '''Replace missing value with mode. Returns value to restore.''' 
    return Mode(column=column)

def CorrCoefRestore(df, row_start, row_end):
    '''Replace missing value with CorrCoef. Returns value to restore.'''
    import pandas as pd
    from Algorithms import CorrelationCoefficient
    df = df.iloc[row_start : row_end]
    inds = df.loc[pd.isna(df).any(1), :].index # Find where is None value
    try: # If inds exists
        inds = inds.tolist()[0]
        if row_start == inds:
            df_cut = df.iloc[inds+1 : row_end]
        if row_end >= inds & inds != row_start:
            df_cut = df.iloc[row_start : inds]
        nan_col = df_cut.columns[df.isnull().any()].tolist()[0] # Get row with NaN
        coef_list = []
        coefs_weights = []
        for column in df.columns:
            if column != nan_col:
                coef_list.append(round(CorrelationCoefficient(df_cut[nan_col], df_cut[column]), 2))
                coefs_weights.append(round(coef_list[-1] * (df[column].iloc[inds] - Mean(df_cut[column])), 2))
        PA = round(Mean(df_cut[nan_col]) + 1/(sum(np.abs(coef_list))) * sum(coefs_weights), 2)
        return PA
    except IndexError:
        return