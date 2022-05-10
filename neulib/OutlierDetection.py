from cmath import inf
from textwrap import indent
import pandas as pd
import numpy as np
from neulib.Algorithms import IsSymmetric, Mean, StdDeviation

def SimpleOutDetect(dataframe, info=True, autorm=False):
    '''Simple algorithm. Remove all outliers from the vector. Returns cleared dataframe is autorm is True.'''

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

def Chauvenet(dataframe, info=True, autorm=False, outlr=[]):
    '''Chauvenet algorithm. Remove all outliers from the vector. Returns cleared dataframe is autorm is True.'''
    # TODO: fix print

    from scipy import special
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning) 

    for column in dataframe:
        if info is True:
            print(f'Checking column: {column}...')
        vector = np.array(dataframe[column])
        i = 0
        flag = 0
        if len(outlr) == 0:
            outliers = []
        else:
            outliers = outlr
            #outliers = list(dict.fromkeys(outlr))
        dictionary = {}
        for elem in vector:
            is_out_cond = special.erfc(np.abs(elem - Mean(vector=vector))/StdDeviation(vector=vector)) < 1 / (2 * len(vector))
            if is_out_cond:
                outliers.append(elem)
                if info is True:
                    print(f'{elem} is outlier!')
                if autorm is True:
                    vector = np.delete(vector, i)
                    condition = dataframe[column] == elem
                    out = dataframe[column].index[condition]
                    dataframe.drop(index=out, inplace = True)
                    i -= 1
                    if info is True:
                        print('Outlier deleted.')
                dictionary.update({column:outliers})
                Chauvenet(dataframe=dataframe, info=info, autorm=autorm, outlr=outliers)
            i += 1
        if dictionary:
            print(f'Detected outliers: {dictionary}')
    return dataframe
