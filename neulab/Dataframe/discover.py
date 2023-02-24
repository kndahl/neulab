import sys
sys.path.append('.')
import warnings

def std_deviation(data, fillna=None):
    """
    Returns a DataFrame containing the std deviation in a given DataFrame.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input DataFrame.
    fillna (optional): 'mode', 'median', 'mean' options.
        The value to use to fill missing values. If None, NaN values are not filled.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the std deviation of DataFrame for each column.
    """
    from neulab.Vector.discover import std_deviation

    df_copy = data.copy()

    if df_copy.isna().any().any():
        warnings.warn('The DataFrame contains NaN values')

    # Apply fillna method for each column
    if fillna is not None:
        for col in df_copy.columns:
            if fillna == 'mode':
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
            elif fillna == 'median':
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            elif fillna == 'mean':
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)

    # Apply std_deviation function for each column
    return df_copy.apply(std_deviation)


def euclidean_matrix(data, fillna=None):
    """
    Returns a DataFrame containing the Euclidean distance between all pairs of columns in a given DataFrame.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The input DataFrame.
    fillna (optional): 'mode', 'median', 'mean' options.
        The value to use to fill missing values. If None, NaN values are not filled.
    
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the Euclidean distance between all pairs of columns in the input DataFrame.
    """

    import pandas as pd
    import numpy as np

    df_copy = data.copy()

    distances = pd.DataFrame(columns=df_copy.columns, index=df_copy.columns)

    if df_copy.isna().any().any():
        warnings.warn('The DataFrame contains NaN values')
    
    for i, col1 in enumerate(df_copy.columns):
        for j, col2 in enumerate(df_copy.columns):
            if i == j:
                distances.loc[col1, col2] = 0
            elif i < j:
                if fillna is not None:
                    if fillna == 'mode':
                        point1 = df_copy[col1].fillna(df_copy[col1].mode().iloc[0])
                        point2 = df_copy[col2].fillna(df_copy[col2].mode().iloc[0])
                    elif fillna == 'median':
                        point1 = df_copy[col1].fillna(df_copy[col1].median())
                        point2 = df_copy[col2].fillna(df_copy[col2].median())
                    elif fillna == 'mean':
                        point1 = df_copy[col1].fillna(df_copy[col1].mean())
                        point2 = df_copy[col2].fillna(df_copy[col2].mean())
                    else:
                        raise ValueError('Invalid fillna_method specified')
                else:
                    point1 = df_copy[col1].values
                    point2 = df_copy[col2].values
            
                # Check that the input vectors have the same shape.
                if point1.shape != point2.shape:
                    raise ValueError("Input vectors must have the same shape")
        
                distance = np.sqrt(np.sum((point1 - point2) ** 2))
                distances.loc[col1, col2] = distance
                distances.loc[col2, col1] = distance
    
    return distances


def manhattan_matrix(data, fillna=None):
    """
    Returns a DataFrame containing the Manhattan distance between all pairs of columns in a given DataFrame.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The input DataFrame.
    fillna (optional): 'mode', 'median', 'mean' options.
        The value to use to fill missing values. If None, NaN values are not filled.
    
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the Manhattan distance between all pairs of columns in the input DataFrame.
    """

    import pandas as pd
    import numpy as np

    df_copy = data.copy()

    distances = pd.DataFrame(columns=df_copy.columns, index=df_copy.columns)

    if df_copy.isna().any().any():
        warnings.warn('The DataFrame contains NaN values')
    
    for i, col1 in enumerate(df_copy.columns):
        for j, col2 in enumerate(df_copy.columns):
            if i == j:
                distances.loc[col1, col2] = 0
            elif i < j:
                if fillna is not None:
                    if fillna == 'mode':
                        point1 = df_copy[col1].fillna(df_copy[col1].mode().iloc[0])
                        point2 = df_copy[col2].fillna(df_copy[col2].mode().iloc[0])
                    elif fillna == 'median':
                        point1 = df_copy[col1].fillna(df_copy[col1].median())
                        point2 = df_copy[col2].fillna(df_copy[col2].median())
                    elif fillna == 'mean':
                        point1 = df_copy[col1].fillna(df_copy[col1].mean())
                        point2 = df_copy[col2].fillna(df_copy[col2].mean())
                    else:
                        raise ValueError('Invalid fillna_method specified')
                else:
                    point1 = df_copy[col1].values
                    point2 = df_copy[col2].values
            
                # Check that the input vectors have the same shape.
                if point1.shape != point2.shape:
                    raise ValueError("Input vectors must have the same shape")
        
                distance = np.sum(np.abs(point1 - point2))
                distances.loc[col1, col2] = distance
                distances.loc[col2, col1] = distance
    
    return distances


def max_matrix(data, fillna=None):
    """
    Returns a DataFrame containing the Max distance between all pairs of columns in a given DataFrame.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The input DataFrame.
    fillna (optional): 'mode', 'median', 'mean' options.
        The value to use to fill missing values. If None, NaN values are not filled.
    
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the Max distance between all pairs of columns in the input DataFrame.
    """

    import pandas as pd
    import numpy as np

    df_copy = data.copy()

    distances = pd.DataFrame(columns=df_copy.columns, index=df_copy.columns)

    if df_copy.isna().any().any():
        warnings.warn('The DataFrame contains NaN values')
    
    for i, col1 in enumerate(df_copy.columns):
        for j, col2 in enumerate(df_copy.columns):
            if i == j:
                distances.loc[col1, col2] = 0
            elif i < j:
                if fillna is not None:
                    if fillna == 'mode':
                        point1 = df_copy[col1].fillna(df_copy[col1].mode().iloc[0])
                        point2 = df_copy[col2].fillna(df_copy[col2].mode().iloc[0])
                    elif fillna == 'median':
                        point1 = df_copy[col1].fillna(df_copy[col1].median())
                        point2 = df_copy[col2].fillna(df_copy[col2].median())
                    elif fillna == 'mean':
                        point1 = df_copy[col1].fillna(df_copy[col1].mean())
                        point2 = df_copy[col2].fillna(df_copy[col2].mean())
                    else:
                        raise ValueError('Invalid fillna_method specified')
                else:
                    point1 = df_copy[col1].values
                    point2 = df_copy[col2].values
            
                # Check that the input vectors have the same shape.
                if point1.shape != point2.shape:
                    raise ValueError("Input vectors must have the same shape")
        
                distance = np.max(np.abs(point1 - point2))
                distances.loc[col1, col2] = distance
                distances.loc[col2, col1] = distance
    
    return distances


def correlation_matrix(data, method='pearson', plot=False):
    """
    Returns a DataFrame containing the correlation matrix between all pairs of columns in a given DataFrame.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The input DataFrame.
    method (optional): str
        The method used to compute the correlation coefficients. Default is 'pearson'.
        Other options are 'kendall' and 'spearman'.
    
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the correlation matrix between all pairs of columns in the input DataFrame.
    """

    df_copy = data.copy()

    corr_matrix = df_copy.corr(method=method)

    if plot:
        display(corr_matrix.style.background_gradient(cmap='coolwarm'))

    return corr_matrix


def check_missing(data, plot=False):
    """
    Check missing values in a DataFrame and returns a DataFrame containing the number and percentage of missing values for each column.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The input DataFrame to check for missing values.
    plot : bool, optional
        If True, a bar plot of the number of missing values for each feature is displayed. Default is False.
    
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the number and percentage of missing values for each column in the input DataFrame that has missing values.
    """

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df_copy = data.copy()

    nan_count = df_copy.isna().sum()
    sorted_columns = nan_count.sort_values(ascending=False)
    pct = sorted_columns.apply(lambda x: (x / df_copy.shape[0]) * 100)
    stat = pd.concat([sorted_columns, pct], axis=1)
    stat = stat.rename(columns={0: 'num', 1: 'pct'})
    if plot == True:
        missing_values = stat.copy()
        missing_values.drop(columns=['pct'], inplace=True)
        missing_values.columns = ['count']
        missing_values.index.names = ['feature']
        missing_values['feature'] = missing_values.index
        sns.set(style="whitegrid", color_codes=True)
        sns.barplot(x = 'feature', y = 'count', data=missing_values)
        plt.xticks(rotation = 90)
        plt.show()
    return stat.loc[stat['num'].gt(0)]


def plot_missing_value_dispersion(data):
    """
    Creates a heatmap showing the locations of missing values in a given DataFrame.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input DataFrame.

    Returns:
    --------
    None
        This function only displays a heatmap and does not return anything.
    """

    import seaborn as sns

    df_copy = data.copy()

    sns.heatmap(df_copy.isnull(), cbar=False)


def plot_categorical(data):
    """
    Plot the number of different values for each categorical feature in the dataframe.

    Parameters
    ----------
    data: Pandas dataframe
    
    Returns
    -------
    None
        This function only displays a heatmap and does not return anything.
    """

    import matplotlib.pyplot as plt

    plt.figure()
    data.nunique().plot.bar()
    plt.title('Number of different values')
    plt.show()

