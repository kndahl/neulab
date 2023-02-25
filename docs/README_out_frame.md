# Z-score method
Function: zscore
### Perform Z-score criterion-based outlier detection for a pandas DataFrame.
Function is used to detect outliers in a pandas DataFrame using the Z-score method. It takes a pandas DataFrame as input, and returns a dictionary with the outliers for each column of the DataFrame. The function can optionally fill in NaN values using one of three methods - mode, median, or mean - before detecting outliers.
## Parameters
data : (pandas DataFrame)
    The input DataFrame for which outliers are to be detected.
fillna (optional): str
    The value to use to fill missing values. It can be 'mode', 'median', 'mean', or None. Default is None.
plot (optional): bool
    Whether to plot the outliers or not. Default is False.
## Returns
outliers (dictionary): A dictionary of outliers.
## Usage
```python
from neulab.Dataframe.outliers import zscore

# Create a sample DataFrame
df = pd.DataFrame({
    'col1': [1, 2, 3, 4, 100, 6, 7, 8, 9, 10, 100],
    'col2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, -100],
    'col3': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10],
    'col4': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
})

# Detect the outliers
outliers = zscore(df)

# Print the outliers
print(outliers)

# Plot the scatter plot with outliers marked in red
zscore(df, plot=True)

Output:
{'col1': [100, 100], 'col2': [-100], 'col3': [10], 'col4': []}
```

# Chauvenet method
Function: chauvenet
### Perform Chauvenet criterion-based outlier detection for a pandas DataFrame.
The function detects outliers in every column of the DataFrame by applying the Chauvenet criterion, which calculates the z-scores of each value and identifies those that fall beyond a certain threshold. The function can optionally fill in NaN values using one of three methods - mode, median, or mean - before detecting outliers.
## Parameters:
data : pandas.DataFrame
    The input DataFrame for which outliers are to be detected.
fillna (optional): str
    The value to use to fill missing values. It can be 'mode', 'median', 'mean', or None. Default is None.
plot (optional): bool
    Whether to plot the outliers or not. Default is False.
## Returns
outliers (dictionary): A dictionary of outliers.
## Usage
```python
import pandas as pd
from neulab.Dataframe.outliers import chauvenet

# Create a sample DataFrame
df = pd.DataFrame({
    'col1': [1, 2, 3, 4, 100, 6, 7, 8, 9, 10, 100],
    'col2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, -100],
    'col3': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10],
    'col4': [1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1]
})

outliers = chauvenet(df, fillna='mean', plot=True)

# Print the outliers
print(outliers)

Output:
{'col1': [100, 100], 'col2': [-100], 'col3': [10], 'col4': [4]}
```

# Quartile method
Function: quartile
### Perform Quartile based outlier detection for a pandas DataFrame.
The function detects outliers in every column of the DataFrame by calculating
the quartiles of each value and identifying those that fall beyond a certain threshold.
The function can optionally fill in NaN values using one of three methods - mode, median, or mean -
before detecting outliers.
## Parameters:
data : pandas.DataFrame
    The input DataFrame for which outliers are to be detected.
fillna (optional): str
    The value to use to fill missing values. It can be 'mode', 'median', 'mean', or None. Default is None.
plot (optional): bool
    Whether to plot the outliers or not. Default is False.
## Returns
outliers (dictionary): A dictionary of outliers.
## Usage
```python
import pandas as pd
from neulab.Dataframe.outliers import quartile


# Create a sample DataFrame
df = pd.DataFrame({
    'col1': [1, 2, 3, 4, 100, 6, 7, 8, 9, 10, 100],
    'col2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, -100],
    'col3': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10]
})

outliers = quartile(df, fillna='mean', plot=True)

Output:
{'col1': [100, 100], 'col2': [-100], 'col3': [10]}
```