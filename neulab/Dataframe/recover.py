import sys
sys.path.append('.')

def simple_imputation(data, method='mean'):
    """
    Perform simple imputation for NaN values in a DataFrame using the specified method.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input DataFrame containing NaN values to be imputed.
    method : str
        The method used to impute the missing values. Valid options are 'mean', 'median', 'mode'.
        Default is 'mean'.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with NaN values imputed using the specified method.
    """
   
    df_copy = data.copy()

    isna_stat = (df_copy.isna().sum() / df_copy.shape[0]).sort_values(ascending=True)
    if isna_stat.max() > 0.0: 
        if method == 'mean':
            df_copy = df_copy.fillna(df_copy.mean())
            print('NaN values imputed with mean.')
        elif method == 'median':
            df_copy = df_copy.fillna(df_copy.median())
            print('NaN values imputed with median.')
        elif method == 'mode':
            df_copy = df_copy.fillna(df_copy.mode().iloc[0])
            print('NaN values imputed with mode.')
        else:
            raise ValueError(f'Invalid imputation method: {method}.')
    else: 
       print('No need to impute data.')
    return df_copy


def distance_imputation(data, metric='euclidean'):
    """
    Perform imputation for NaN values in a DataFrame using the distance method.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input DataFrame containing NaN values to be imputed.
    metric : str
        The metric used to impute the missing values. Valid options are 'euclidean', 'manhattan', or 'max'.
        Default is 'euclidean'.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with NaN values imputed using the specified method.
    """


    from neulab.Vector.recover import replace_missing_with_distance

    if metric not in ['euclidean', 'manhattan', 'max']:
        raise ValueError(f'Invalid imputation metric: {metric}.')

    df_copy = data.copy()

    isna_stat = (df_copy.isna().sum() / df_copy.shape[0]).sort_values(ascending=True)
    if isna_stat.max() > 0.0: 

        vector = df_copy.values.T

        recovered_array = replace_missing_with_distance(vector, metric=metric)

        # replace NaN values in df_copy with the recovered values
        df_copy.iloc[:, :] = recovered_array[0].T

        print(f'NaN values imputed with {metric} metric.')

    else: 
       print('No need to impute data.')

    return df_copy


def iterative_imputation(data):
    """
    Imputes missing values in a given DataFrame using the IterativeImputer.

    The algorithm works by modeling the missing values as 
    a function of the other features in the dataset. 
    It starts by filling in missing values with initial estimates, 
    such as the mean or median of the non-missing values. 
    It then uses this initial estimate to fit a model to the non-missing 
    values and predict the missing values. This process is repeated multiple 
    times, with the predicted values from each iteration being used as the 
    input for the next iteration.

    IterativeImputer is useful when the missing values in a dataset are not 
    completely at random, and there is some pattern or relationship between 
    the missing values and the other features. However, it is important to 
    note that this method can be computationally expensive, especially for 
    large datasets or when using complex regression models.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The input DataFrame.
    
    Returns:
    --------
    pandas.DataFrame
        A DataFrame with missing values imputed using IterativeImputer, if any.
    """

    import pandas as pd
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    df_copy = data.copy()

    isna_stat = (df_copy.isna().sum()/df_copy.shape[0]).sort_values(ascending=True) 
    if isna_stat.max() > 0.0: 
       print('Imputing NaN using IterativeImputer') 
       df_copy = pd.DataFrame(IterativeImputer(random_state=0).fit_transform(df_copy), columns = df_copy.columns)  
    else: 
       print('No need to impute data.')
    return df_copy
