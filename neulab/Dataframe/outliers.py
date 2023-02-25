
def zscore(data, fillna=None, plot=False):
    """
    Perform z-score based outlier detection for a pandas DataFrame.
    It detects outliers in every column of the DataFrame by calculating 
    the z-scores of each value and identifying those that fall beyond a certain threshold. 
    The function can optionally fill in NaN values using one of three methods - mode, median, or mean - 
    before detecting outliers.

    Parameters:
    data : pandas.DataFrame
    The input DataFrame for which outliers are to be detected.
    fillna (optional): 'mode', 'median', 'mean' options.
    The value to use to fill missing values. If None, NaN values are not filled.
    plot (optional): bool.
    Whether to plot the outliers or not. Default is False.

    Returns:
    dict: A dictionary of outliers.
    """
    
    import pandas as pd
    import numpy as np

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    if not data.select_dtypes(include=[np.number]).columns.any():
        raise ValueError("Input DataFrame must contain numerical data.")

    df_copy = data.copy()

    from neulab.Vector.outliers import zscore_outliers

    if fillna is not None:
        if fillna not in ['mode', 'median', 'mean']:
            raise ValueError('Invalid fillna method specified.')

    # Get outliers of every column
    outliers = {}
    for col in df_copy.columns:
        # Apply fillna method for each column
        if fillna is not None:
            if fillna == 'mode':
                _, col_outliers = zscore_outliers(df_copy[col].fillna(df_copy[col].mode()[0]))
            elif fillna == 'median':
                _, col_outliers = zscore_outliers(df_copy[col].fillna(df_copy[col].median()))
            elif fillna == 'mean':
                _, col_outliers = zscore_outliers(df_copy[col].fillna(df_copy[col].mean()))
        else:
            _, col_outliers = zscore_outliers(df_copy[col])

        outliers[col] = col_outliers

    if plot:
        plot_outliers(data, outliers)

    return outliers


def chauvenet(data, fillna=None, plot=False):
    """
    Perform chauvenet based outlier detection for a pandas DataFrame.
    It detects outliers in every column of the DataFrame by applying 
    the Chauvenet criterion, which calculates the z-scores of each value 
    and identifies those that fall beyond a certain threshold. 
    The function can optionally fill in NaN values using one of three methods - mode,
    median, or mean - before detecting outliers.

    Parameters:
    data : pandas.DataFrame
    The input DataFrame for which outliers are to be detected.
    fillna (optional): 'mode', 'median', 'mean' options.
    The value to use to fill missing values. If None, NaN values are not filled.
    plot (optional): bool.
    Whether to plot the outliers or not. Default is False.

    Returns:
    dict: A dictionary of outliers.
    """

    import pandas as pd
    import numpy as np

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    if not data.select_dtypes(include=[np.number]).columns.any():
        raise ValueError("Input DataFrame must contain numerical data.")

    df_copy = data.copy()

    from neulab.Vector.outliers import chauvenet_outliers

    if fillna is not None:
        if fillna not in ['mode', 'median', 'mean']:
            raise ValueError('Invalid fillna method specified.')

    # Get outliers of every column
    outliers = {}
    for col in df_copy.columns:
        # Apply fillna method for each column
        if fillna is not None:
            if fillna == 'mode':
                _, col_outliers = chauvenet_outliers(df_copy[col].fillna(df_copy[col].mode()[0]))
            elif fillna == 'median':
                _, col_outliers = chauvenet_outliers(df_copy[col].fillna(df_copy[col].median()))
            elif fillna == 'mean':
                _, col_outliers = chauvenet_outliers(df_copy[col].fillna(df_copy[col].mean()))
        else:
            _, col_outliers = chauvenet_outliers(df_copy[col])

        outliers[col] = list(col_outliers)

    if plot:
        plot_outliers(data, outliers)

    return outliers


def quartile(data, fillna=None, plot=False):
    """
    Perform quartile-based outlier detection for a pandas DataFrame.
    It detects outliers in every column of the DataFrame by calculating
    the quartiles of each value and identifying those that fall beyond a certain threshold.
    The function can optionally fill in NaN values using one of three methods - mode, median, or mean -
    before detecting outliers.

    Parameters:
    data : pandas.DataFrame
    The input DataFrame for which outliers are to be detected.
    fillna (optional): 'mode', 'median', 'mean' options.
    The value to use to fill missing values. If None, NaN values are not filled.
    plot (optional): bool.
    Whether to plot the outliers or not. Default is False.

    Returns:
    dict: A dictionary of outliers.
    """

    import pandas as pd
    import numpy as np

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    if not data.select_dtypes(include=[np.number]).columns.any():
        raise ValueError("Input DataFrame must contain numerical data.")

    df_copy = data.copy()

    from neulab.Vector.outliers import quartile_outliers

    if fillna is not None:
        if fillna not in ['mode', 'median', 'mean']:
            raise ValueError('Invalid fillna method specified.')

    # Get outliers of every column
    outliers = {}
    for col in df_copy.columns:
        # Apply fillna method for each column
        if fillna is not None:
            if fillna == 'mode':
                _, col_outliers = quartile_outliers(df_copy[col].fillna(df_copy[col].mode()[0]))
            elif fillna == 'median':
                _, col_outliers = quartile_outliers(df_copy[col].fillna(df_copy[col].median()))
            elif fillna == 'mean':
                _, col_outliers = quartile_outliers(df_copy[col].fillna(df_copy[col].mean()))
        else:
            _, col_outliers = quartile_outliers(df_copy[col])

        outliers[col] = list(col_outliers)

    if plot:
        plot_outliers(data, outliers)

    return outliers


def plot_outliers(data, outliers):
    """
    Plot outliers for each column in a given pandas dataframe.

    Parameters
    ----------
    data : pandas DataFrame
        The input data to plot.

    outliers: dictionary
        A dictionary with information about oultliers

    Returns
    -------
    None
        This function only displays the plot and does not return anything.
    """

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    if not data.select_dtypes(include=[np.number]).columns.any():
        raise ValueError("Input DataFrame must contain numerical data.")

    df_copy = data.copy()

    num_values = len([v for k, v in outliers.items() if len(v) > 0])

    if num_values > 0:

        # Create subplots for each column
        fig, axs = plt.subplots(nrows=num_values, ncols=1, figsize=(6, 4*len(df_copy.columns)))

        y = 0
        for i, col in enumerate(df_copy.columns):
            if any(df_copy[col].isin(outliers[col]) == True):

                ax = axs[y]

                # Plot the non-outliers
                non_outliers = df_copy[~df_copy[col].isin(outliers[col])]
                ax.scatter(non_outliers.index, non_outliers[col], label=col, alpha=0.7, color='green')

                # Plot the outliers
                col_outliers = df_copy[df_copy[col].isin(outliers[col])]
                ax.scatter(col_outliers.index, col_outliers[col], marker='x', label=f"{col} (outliers)", color='red')

                # Add legend and labels
                ax.legend()
                ax.set_xlabel('Index')
                ax.set_ylabel('Value')

                # Set title for each subplot
                ax.set_title(f"{col} Distribution")
                y += 1

        plt.tight_layout()
        plt.show()

    else:
        print('There are no outliers to plot.')