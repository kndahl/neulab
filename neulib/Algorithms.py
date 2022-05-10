import numpy as np
from scipy.spatial.distance import cdist
from scipy import stats

def CorrelationCoefficient(vector1, vector2):
    '''Returns correlation coef between column1 and column2.'''

    v1 = np.array(vector1)
    v2 = np.array(vector2)

    cor_coef = np.corrcoef(v1, v2)
    return cor_coef[0][1]

def EuclidMertic(vector1, vector2):
    '''Returns Euclidean distance between vector1 and vector2.'''

    point_1 = np.array(vector1)
    point_2 = np.array(vector2)

    point_1 = point_1.reshape(1, -1)
    point_2 = point_2.reshape(1, -1)

    euclid_metric = cdist(point_1, point_2, 'euclidean')
    return euclid_metric[0][0]

def ManhattanMetric(vector1, vector2):
    '''Returns Manhatta distance between column1 and column2.'''

    point_1 = np.array(vector1)
    point_2 = np.array(vector2)

    point_1 = point_1.reshape(1, -1)
    point_2 = point_2.reshape(1, -1)

    manhattan_metric = cdist(point_1, point_2, metric='cityblock')
    return manhattan_metric[0][0]

def MaxMetric(vector1, vector2):
    '''Returns MaxMetric distance between column1 and column2.'''
    
    point_1 = np.array(vector1)
    point_2 = np.array(vector2)

    max_metric = max(np.abs(point_1-point_2))
    return max_metric

def Mean(vector):
    '''Replace missing value with mean. Returns value to restore.'''

    vector = np.array(vector)
    vector_mean = vector.mean()
    return vector_mean

def Median(vector):
    '''Replace missing value with median. Returns value to restore.'''

    vector = np.array(vector)
    vector_median = np.median(vector)
    return vector_median

def Mode(vector):
    '''Replace missing value with mode. Returns value to restore.''' 

    vector = np.array(vector)
    mode_vector = stats.mode(vector)
    return mode_vector[0][0]

def StdDeviation(vector):
    '''Calculates standart deviation.'''

    vector = np.array(vector)
    std = np.std(vector, ddof=1)
    return std


def IsSymmetric(vector):
    '''Detects if vector is symmetric or asymmetric. Returns True if vector is symmetric or False if vector is asymmetric.'''

    vectype = np.abs(Median(vector=vector) - Mean(vector=vector)) <= 3 * np.sqrt((StdDeviation(vector=vector) ** 2) / len(vector))
    return vectype