import numpy as np
from scipy import stats

def MeanRestore(column):
    '''Replace missing value with mean value. Returns value to restore.'''

    vector = np.array(column)
    vector_mean = vector.mean()
    return vector_mean

def MedianRestore(column):
    '''Replace missing value with median value. Returns value to restore.'''

    vector = np.array(column)
    vector_median = np.median(vector)
    return vector_median

def ModeRestore(column):
    '''Replace missing value with mode value. Returns value to restore.''' 

    vector = np.array(column)
    mode_vector = stats.mode(vector)
    return mode_vector[0][0]
