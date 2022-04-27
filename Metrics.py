import numpy as np
from scipy.spatial.distance import cdist

def EuclidMertic(column1, column2):
    '''Restore missing value by Euclidean Metric. Returns value to restore.'''

    point_1 = np.array(column1)
    point_2 = np.array(column2)

    point_1 = point_1.reshape(1, -1)
    point_2 = point_2.reshape(1, -1)

    euclid_metric = cdist(point_1, point_2, 'euclidean')
    return euclid_metric[0][0]

def ManhattanMetric(column1, column2):
    '''Restore missing value by Manhattan Metric. Returns value to restore.'''

    point_1 = np.array(column1)
    point_2 = np.array(column2)

    point_1 = point_1.reshape(1, -1)
    point_2 = point_2.reshape(1, -1)

    manhattan_metric = cdist(point_1, point_2, metric='cityblock')
    return manhattan_metric[0][0]

def MaxMetric(column1, column2):
    '''Restore missing value by Max Metric. Returns value to restore.'''
    
    point_1 = np.array(column1)
    point_2 = np.array(column2)

    max_metric = max(np.abs(point_1-point_2))
    return max_metric