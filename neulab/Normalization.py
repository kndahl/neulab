import numpy as np

def InterNormalization(column):
    ''' Normalization function that normalize data in column by Min and Max algotithm: [0, 1]. '''

    vector = np.array(column)
    inter_normalization = (vector - min(vector))/(max(vector) - min(vector))
    return inter_normalization

def MeanNormalization(column):
    ''' Normalization function that normalize data in column by mean algotithm '''

    vector = np.array(column)
    mean_normalization = (vector - np.mean(vector))/(np.std(vector))
    return mean_normalization
