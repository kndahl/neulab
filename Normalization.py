import numpy as np

def MaxMinNormalization(column):
    ''' Normalization function that normalize data in column by Min and Max algotithm '''

    vector = np.array(column)
    maxmin_normalization = (vector - min(vector))/max(vector) - min(vector)
    return maxmin_normalization

def MeanNormalization(column):
    ''' Normalization function that normalize data in column by mean algotithm '''

    vector = np.array(column)
    mean_normalization = (vector - np.mean(vector))/np.std(vector)
    return mean_normalization
