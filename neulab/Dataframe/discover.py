
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
    import pandas as pd
    import warnings

    if isinstance(data, pd.DataFrame):
        df_copy = data.copy()
    else:
        raise TypeError("Input data must be a pandas DataFrame.")

    if fillna is not None:
        if fillna not in ['mode', 'median', 'mean']:
            raise ValueError('Invalid fillna method specified.')

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
    import warnings

    if isinstance(data, pd.DataFrame):
        df_copy = data.copy()
    else:
        raise TypeError("Input data must be a pandas DataFrame.")

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
                        raise ValueError('Invalid fillna method specified')
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
    import warnings

    if isinstance(data, pd.DataFrame):
        df_copy = data.copy()
    else:
        raise TypeError("Input data must be a pandas DataFrame.")

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
    import warnings

    if isinstance(data, pd.DataFrame):
        df_copy = data.copy()
    else:
        raise TypeError("Input data must be a pandas DataFrame.")

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

    import pandas as pd

    if isinstance(data, pd.DataFrame):
        df_copy = data.copy()
    else:
        raise TypeError("Input data must be a pandas DataFrame.")

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

    if isinstance(data, pd.DataFrame):
        df_copy = data.copy()
    else:
        raise TypeError("Input data must be a pandas DataFrame.")

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
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        df_copy = data.copy()
    else:
        raise TypeError("Input data must be a pandas DataFrame.")

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
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        pass
    else:
        raise TypeError("Input data must be a pandas DataFrame.")

    plt.figure()
    data.nunique().plot.bar()
    plt.title('Number of different values')
    plt.show()


def plot_scatter(data):
    """
    Plots scatter plots for each column in a given pandas dataframe.

    Parameters
    ----------
    data: pandas DataFrame
        The input data to plot.

    Returns
    -------
    None
        This function only displays the plot and does not return anything.
    """

    import matplotlib.pyplot as plt
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        df_copy = data.copy()
    elif isinstance(data, pd.Series):
        df_copy = pd.DataFrame(data)
    else:
        raise TypeError("Input data must be a pandas DataFrame or Series.")

    # Create subplots for each column
    fig, axs = plt.subplots(nrows=len(df_copy.columns), ncols=1, figsize=(6, 4*len(df_copy.columns)))

    # Convert axs to a list if it is a single AxesSubplot object
    if isinstance(axs, plt.Axes):
        axs = [axs]

    for i, col in enumerate(df_copy.columns):

        ax = axs[i]

        # Plot
        ax.scatter(df_copy[col].index, df_copy[col], label=col, alpha=0.7, color='green')

        # Add legend and labels
        ax.legend()
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')

        # Set title for each subplot
        ax.set_title(f"{col} Distribution")

    plt.tight_layout()
    plt.show()


def get_categorical_columns(data, threshold=0.05):
    """
    Detects categorical columns in a pandas DataFrame.

    Parameters
    ----------
    data: pandas DataFrame
        The input data to analyze.
    threshold: int or float, optional
        The proportion of unique values below which a column is considered categorical,
        by default 0.05.

    Returns
    -------
    list:
        A list of column names that are likely to be categorical.
    """

    import pandas as pd

    if isinstance(data, pd.DataFrame):
        df_copy = data.copy()
    else:
        raise TypeError("Input data must be a pandas DataFrame.")

    # Define a threshold for the number of unique values relative to column size
    unique_value_threshold = threshold

    # Initialize an empty list to store categorical column names
    categorical_columns = []

    for col in df_copy.columns:
        # Count the number of unique values in the column
        num_unique_values = len(df_copy[col].unique())

        # Compute the proportion of unique values relative to column size
        unique_value_proportion = num_unique_values / len(df_copy)

        # Check if the proportion is below the threshold
        if unique_value_proportion < unique_value_threshold:
            categorical_columns.append(col)

    return categorical_columns
