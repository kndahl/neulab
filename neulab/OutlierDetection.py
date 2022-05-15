from cmath import inf
from textwrap import indent
from bleach import clean
import pandas as pd
import numpy as np

def SimpleOutDetect(dataframe, info=True, autorm=False):
    '''Simple algorithm. Remove all outliers from the vector. Returns cleared dataframe is autorm is True.'''

    from neulab.Algorithms import IsSymmetric, Mean, StdDeviation
    for column in dataframe:
        vector = np.array(dataframe[column])
        # Define vector type
        if IsSymmetric(vector=vector):
            if info is True:
                print(f'Vector {column} is symmetric.')
            i = 0
            outliers = []
            dict = {}
            for elem in vector:
                cleared = np.delete(vector, i)
                mean = Mean(vector=cleared)
                std = StdDeviation(vector=cleared)
                interval1 = mean - 3 * std
                interval2 = mean + 3 * std
                if interval1 < elem < interval2:
                    pass
                else:
                    outliers.append(elem)
                    if info is True:
                        print(f'Found outlier: {elem}')
                    if autorm is True:
                        vector = np.delete(vector, i)
                        condition = dataframe[column] == elem
                        out = dataframe[column].index[condition]
                        dataframe.drop(index=out, inplace = True)
                        i -= 1
                    dict.update({column:outliers})
                i += 1
        else:
            if info is True:
                print(f'Vector {column} is asymmetric.')
            i = 0
            outliers = []
            dict = {}
            for elem in vector:
                cleared = np.delete(vector, i)
                mean = Mean(vector=cleared)
                std = StdDeviation(vector=cleared)
                interval1 = mean - 3 * std
                interval2 = mean + 3 * std
                if interval1 < elem < interval2:
                    pass
                else:
                    outliers.append(elem)
                    if info is True:
                        print(f'Found outlier: {elem}')
                    if autorm is True:
                        vector = np.delete(vector, i)
                        condition = dataframe[column] == elem
                        out = dataframe[column].index[condition]
                        dataframe.drop(index=out, inplace = True)
                        if info is True:
                            print('Outlier deleted.')
                        i -= 1
                    dict.update({column:outliers})
                i += 1

        if dict:
            print(f'Detected outliers: {dict}')

    return dataframe

def Chauvenet(dataframe, info=True, autorm=False):
    '''Chauvenet algorithm. Remove all outliers from the vector. Returns cleared dataframe is autorm is True.'''

    from scipy import special
    from neulab.Algorithms import Mean, StdDeviation
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning) 

    def loop(i, elem, vector, outliers, dataframe, dictionary):
        if len(outliers) == 0:
            outliers = []
        else:
            outliers = list(dict.fromkeys(outliers))
        for elem in vector:
            is_out_cond = special.erfc(np.abs(elem - Mean(vector=vector))/StdDeviation(vector=vector)) < 1 / (2 * len(vector))
            if is_out_cond:
                outliers.append(elem)
                if autorm is True:
                    vector = np.delete(vector, i)
                    condition = dataframe[column] == elem
                    out = dataframe[column].index[condition]
                    dataframe.drop(index=out, inplace = True)
                    i -= 1
                dictionary.update({column:outliers})
                loop(i=0, elem=elem, vector=vector, outliers=outliers, dataframe=dataframe, dictionary=dictionary)
            i += 1

    for column in dataframe:
        if info is True:
            print(f'Checking column: {column}...')
        vector = np.array(dataframe[column])
        i = 0
        outliers = []
        dictionary = {}
        for elem in vector:
            is_out_cond = special.erfc(np.abs(elem - Mean(vector=vector))/StdDeviation(vector=vector)) < 1 / (2 * len(vector))
            if is_out_cond:
                outliers.append(elem)
                if autorm is True:
                    vector = np.delete(vector, i)
                    condition = dataframe[column] == elem
                    out = dataframe[column].index[condition]
                    dataframe.drop(index=out, inplace = True)
                    i -= 1
                dictionary.update({column:outliers})
                loop(i=0, elem=elem, vector=vector, outliers=outliers, dataframe=dataframe, dictionary=dictionary)
            i += 1
        if dictionary:
            print(f'Detected outliers: {dictionary}')
    return dataframe

def Quratile(dataframe, info=True, autorm=False):
    '''Quratile algorithm doest use standart deviation and average mean. Remove all outliers from the vector. Returns cleared dataframe is autorm is True.'''

    from neulab.Algorithms import Median
    dictionary = {}
    for column in dataframe:
        i = 0
        outliers = []
        if info is True:
            print(f'Checking column: {column}...')
        vector = np.array(dataframe[column])
        q50 = np.quantile(vector, 0.5)
        q25 = np.quantile(vector, 0.25)
        q75 = np.quantile(vector, 0.75)
        interval1 = q25 - 1.5 * (q75 - q25)
        interval2 = q75 + 1.5 * (q75 - q25)
        if info is True:
            print(f'Q25 = {q25}, Q50 = {q50}, Q75 = {q75}. Interval1 = {interval1}, Interval2 = {interval2}')
        for elem in vector:
            if interval1 < elem < interval2:
                pass
            else:
                outliers.append(elem)
                if autorm is True:
                    vector = np.delete(vector, i)
                    condition = dataframe[column] == elem
                    out = dataframe[column].index[condition]
                    dataframe.drop(index=out, inplace = True)
                    i -= 1
            i += 1
        dictionary.update({column:outliers})
    if dictionary:
        print(f'Detected outliers: {dictionary}')
    return dataframe

def DistQuant(dataframe, metric='euclid', filter='quantile', info=True, autorm=False):
    '''An outlier search algorithm using metrics. The metrics calculate the distance between features and then filter using the quantile algorithm. Returns cleared dataframe is autorm is True.'''
    
    from neulab.Algorithms import EuclidMertic, ManhattanMetric, MaxMetric, Mean, Median, StdDeviation

    indexes = dataframe.index.to_list()
    row_list = []
    row_dict = {}
    dist_dict = {}

    lenght = len(indexes)
    for elem in range(lenght):
        vector = np.array(dataframe.loc[indexes[elem]])
        row_list.append(vector)

    i = 0
    for inx in indexes:
        row_dict.update({inx:row_list[i]})
        i += 1

    for i1, i2 in row_dict.items():
        dist = 0
        for a in row_dict.items():
            if metric == 'euclid':
                dist += EuclidMertic(vector1=i2, vector2=a[1])
            if metric == 'manhattan':
                dist += ManhattanMetric(vector1=i2, vector2=a[1])
            if metric == 'max':
                dist += MaxMetric(vector1=i2, vector2=a[1])
        dist_dict.update({i1:dist})
    if info is True:
        print(f'Distances: {dist_dict}')

    # Find outliers in list of distances (quantile algorithm)
    if filter == 'quantile':
        vector = np.array(list(dist_dict.values()))

        def quant_loop(vector):
            flag = False
            q25 = np.quantile(vector, 0.25)
            q75 = np.quantile(vector, 0.75)
            interval1 = q25 - 1.5 * (q75 - q25)
            interval2 = q75 + 1.5 * (q75 - q25)
            i = 0
            indexes = dataframe.index.tolist()
            for elem in vector:
                if interval1 < elem < interval2:
                    pass
                else:
                    flag = True
                    if info is True:
                        print(f'Detected outlier: \n{dataframe.loc[[indexes[i]]]}')
                    inx = np.where(vector == elem)
                    vector = np.delete(vector, inx)
                    if autorm is True:
                        dataframe.drop(indexes[i], inplace=True)
                i += 1
            return vector, flag
        # Repeat algorithm
        while True and len(dataframe.index) > 2:
            vector, flg = quant_loop(vector=vector)
            if flg is False:
                break

    return dataframe

def DixonTest(dataframe, q=95, info=True, autorm=False):
    '''Dixon Q Test algorithm. Remove all outliers from the vector. Returns cleared dataframe is autorm is True.'''

    q90 = [0.941, 0.765, 0.642, 0.56, 0.507, 0.468, 0.437,
        0.412, 0.392, 0.376, 0.361, 0.349, 0.338, 0.329,
        0.32, 0.313, 0.306, 0.3, 0.295, 0.29, 0.285, 0.281,
        0.277, 0.273, 0.269, 0.266, 0.263, 0.26]

    q95 = [0.97, 0.829, 0.71, 0.625, 0.568, 0.526, 0.493, 0.466,
        0.444, 0.426, 0.41, 0.396, 0.384, 0.374, 0.365, 0.356,
        0.349, 0.342, 0.337, 0.331, 0.326, 0.321, 0.317, 0.312,
        0.308, 0.305, 0.301, 0.29]

    q99 = [0.994, 0.926, 0.821, 0.74, 0.68, 0.634, 0.598, 0.568,
        0.542, 0.522, 0.503, 0.488, 0.475, 0.463, 0.452, 0.442,
        0.433, 0.425, 0.418, 0.411, 0.404, 0.399, 0.393, 0.388,
        0.384, 0.38, 0.376, 0.372]

    Q90 = {n:q for n,q in zip(range(3,len(q90)+1), q90)}
    Q95 = {n:q for n,q in zip(range(3,len(q95)+1), q95)}
    Q99 = {n:q for n,q in zip(range(3,len(q99)+1), q99)}

    def dixon_test(data, q_dict=Q95):

        sdata = sorted(data)
        Q_mindiff, Q_maxdiff = (0,0), (0,0)
        Q_min = (sdata[1] - sdata[0])
        try:
            Q_min /= (sdata[-1] - sdata[0])
        except ZeroDivisionError:
            pass
        Q_mindiff = (Q_min - q_dict[len(data)], sdata[0])
        Q_max = abs((sdata[-2] - sdata[-1]))
        try:
            Q_max /= abs((sdata[0] - sdata[-1]))
        except ZeroDivisionError:
            pass
        Q_maxdiff = (Q_max - q_dict[len(data)], sdata[-1])

        if not Q_mindiff[0] > 0 and not Q_maxdiff[0] > 0:
            outliers = []
        elif Q_mindiff[0] == Q_maxdiff[0]:
            outliers = [Q_mindiff[1], Q_maxdiff[1]]
        elif Q_mindiff[0] > Q_maxdiff[0]:
            outliers = [Q_mindiff[1]]
        else:
            outliers = [Q_maxdiff[1]]

        return outliers

    for column in dataframe:
        vector = np.array(dataframe[column])
        if q == 90:
            outliers = dixon_test(vector, q_dict=Q90)
        if q == 95:
            outliers = dixon_test(vector, q_dict=Q95)
        if q == 99:
            outliers = dixon_test(vector, q_dict=Q99)
        if len(outliers) > 0:
            for elem in outliers:
                condition = dataframe[column] == elem
                out = dataframe[column].index[condition]
                if info is True:
                    print(f'Detected outlier: \n{dataframe.loc[[out[0]]]}')
                if autorm is True:
                    dataframe.drop(index=out, inplace = True)

    return dataframe