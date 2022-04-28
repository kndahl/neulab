import numpy as np
import pandas as pd

def MeanRestore(column):
    '''Replace missing value with mean. Returns value to restore.'''
    from neulib.Algorithms import Mean
    return Mean(column=column)

def MedianRestore(column):
    '''Replace missing value with median. Returns value to restore.'''
    from neulib.Algorithms import Median
    return Median(column=column)

def ModeRestore(column):
    '''Replace missing value with mode. Returns value to restore.''' 
    from neulib.Algorithms import Mode
    return Mode(column=column)

def CorrCoefRestore(df, row_start, row_end):
    '''Replace missing value with CorrCoef value. Returns value to restore.'''
    from neulib.Algorithms import CorrelationCoefficient, Mean
    df = df.iloc[row_start : row_end]
    inds = df.loc[pd.isna(df).any(1), :].index # Find where is None value
    try: # If inds exists
        inds = inds.tolist()[0]
        df_cut = df.drop(inds) # Drop column with NaN
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

def MetricRestore(df, row_start, row_end, metric):
    '''Replace missing value with Metric value. Returns value to restore.'''
    from neulib.Algorithms import EuclidMertic, ManhattanMetric, MaxMetric
    df = df.iloc[row_start : row_end+1]
    inds_col = df.columns[df.isna().any()].tolist() # Find where is None value (col)
    inds_row = df.loc[pd.isna(df).any(1), :].index[0] # Find where is None value (row)
    if len(inds_col) > 1:
        return
    nan_col = df[inds_col]
    df_cut = df.drop(columns=inds_col[0]) # Drop column with NaN
    dist_list = []
    if row_start == inds_row: # If nan at top
        val_list = df[nan_col.columns[0]].iloc[inds_row+1:row_end].tolist()
    else:
        val_list = df[nan_col.columns[0]].iloc[:inds_row].tolist()
    for index, row in df_cut.iterrows():
        if index != inds_row:
            if metric == 'euclid':
                dist_list.append(round(EuclidMertic(vector1=df_cut.iloc[inds_row], vector2=row), 2))
            if metric == 'manhattan':
                dist_list.append(round(ManhattanMetric(vector1=df_cut.iloc[inds_row], vector2=row), 2))
            if metric == 'max':
                dist_list.append(round(MaxMetric(vector1=df_cut.iloc[inds_row], vector2=row), 2))
    DISTANCE = round(1/(sum([1/i for i in dist_list])) * sum([v/d for v, d in zip(val_list, dist_list)]), 2)
    return DISTANCE
