
def min_max_normalize(data, cols_to_normalize):
    """
    Perform Min-Max normalization on the selected columns of the given pandas DataFrame.

    Parameters:
    data (pandas.DataFrame): The DataFrame to normalize.
    cols_to_normalize (list): The list of column names to normalize.

    Returns:
    pandas.DataFrame: The normalized DataFrame.
    """

    import pandas as pd
    import numpy as np

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    df = data.copy()

    for col in cols_to_normalize:
        col_min = df[col].min()
        col_max = df[col].max()
        if np.isnan(col_min) or np.isnan(col_max):
            # Handle NaN values by skipping the normalization
            continue
        df[col] = (df[col] - col_min) / (col_max - col_min)
    return df

def z_score_normalize(data, cols_to_normalize):
    """
    Perform Z-score normalization on the selected columns of the given pandas DataFrame.

    Parameters:
    data (pandas.DataFrame): The DataFrame to normalize.
    cols_to_normalize (list): The list of column names to normalize.

    Returns:
    pandas.DataFrame: The normalized DataFrame.
    """

    import pandas as pd
    import numpy as np

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    df = data.copy()
    
    # Normalize the input feature columns
    for col in cols_to_normalize:
        col_mean = df[col].mean()
        col_std = df[col].std()
        if np.isnan(col_mean) or np.isnan(col_std):
            # Handle NaN values by skipping the normalization
            continue
        df[col] = (df[col] - col_mean) / col_std

    return df