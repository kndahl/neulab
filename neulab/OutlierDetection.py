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
            for elem in vector:
                if interval1 < elem < interval2:
                    pass
                else:
                    flag = True
                    if info is True:
                        print(f'Detected outlier: \n{dataframe.loc[[i]]}')
                    inx = np.where(vector == elem)
                    vector = np.delete(vector, inx)
                    if autorm is True:
                        dataframe.drop(i, inplace=True)
                i += 1
            return vector, flag
        # Repeat algorithm
        while True:
            vector, flg = quant_loop(vector=vector)
            if flg is False:
                break

    return dataframe
