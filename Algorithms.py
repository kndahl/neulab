import numpy as np
from scipy.spatial.distance import cdist
from scipy import stats

def CorrelationCoefficient(column1, column2):
    '''Returns correlation coef between column1 and column2.'''

    col1 = np.array(column1)
    col2 = np.array(column2)

    cor_coef = np.corrcoef(col1, col2)
    return cor_coef[0][1]

def EuclidMertic(column1, column2):
    '''Returns Euclidean distance between column1 and column2.'''

    point_1 = np.array(column1)
    point_2 = np.array(column2)

    point_1 = point_1.reshape(1, -1)
    point_2 = point_2.reshape(1, -1)

    euclid_metric = cdist(point_1, point_2, 'euclidean')
    return euclid_metric[0][0]

def ManhattanMetric(column1, column2):
    '''Returns Manhatta distance between column1 and column2.'''

    point_1 = np.array(column1)
    point_2 = np.array(column2)

    point_1 = point_1.reshape(1, -1)
    point_2 = point_2.reshape(1, -1)

    manhattan_metric = cdist(point_1, point_2, metric='cityblock')
    return manhattan_metric[0][0]

def MaxMetric(column1, column2):
    '''Returns MaxMetric distance between column1 and column2.'''
    
    point_1 = np.array(column1)
    point_2 = np.array(column2)

    max_metric = max(np.abs(point_1-point_2))
    return max_metric

def Mean(column):
    '''Replace missing value with mean. Returns value to restore.'''

    vector = np.array(column)
    vector_mean = vector.mean()
    return vector_mean

def Median(column):
    '''Replace missing value with median. Returns value to restore.'''

    vector = np.array(column)
    vector_median = np.median(vector)
    return vector_median

def Mode(column):
    '''Replace missing value with mode. Returns value to restore.''' 

    vector = np.array(column)
    mode_vector = stats.mode(vector)
    return mode_vector[0][0]