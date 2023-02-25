# Standard deviation
Function: std_deviation
### Function computes the standard deviation of a given pandas DataFrame using the neulab.Vector.discover.std_deviation() function. The function also includes an option to fill missing values in the DataFrame before computing the standard deviation.
## Parameters
data: a pandas DataFrame containing the input data.
fillna (optional): a string indicating the method to use to fill missing values. The options are 'mode', 'median', or 'mean'. If this parameter is not provided or set to None, then the function will not fill missing values in the DataFrame.
## Returns
pandas.DataFrame: A pandas DataFrame containing the standard deviation of each column in the input DataFrame.
## Example Usage
```python
import pandas as pd
from neulab.Dataframe.discover import std_deviation

# create a sample DataFrame with missing values
data = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [1, 2, 3, None, 5],
    'C': [1, 2, 3, 4, 5]
})

# compute the standard deviation with missing values filled with the column mean
result_filled = std_deviation(data, fillna='mean')
```

# Euclidean matrix
Function: euclidean_martix
### Function computes the Euclidean distance between all pairs of columns in a given pandas DataFrame. The Euclidean distance is the length of the straight line between two points in a Euclidean space, which is the square root of the sum of the squared differences between the corresponding elements of the two points.
## Parameters
data: a pandas DataFrame containing the input data.
fillna (optional): a string indicating the method to use to fill missing values. The options are 'mode', 'median', or 'mean'. If this parameter is not provided or set to None, then the function will not fill missing values in the DataFrame.
## Returns
pandas.DataFrame: A pandas DataFrame containing the Euclidean distance between all pairs of columns in the input DataFrame.

## Example Usage
```python
import pandas as pd
from neulab.Dataframe.discover import euclidean_martix

# create a sample DataFrame with missing values
data = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [1, 2, 3, None, 5],
    'C': [1, 2, 3, 4, 5]
})

# compute the Euclidean distance matrix with missing values filled with the column mean
result_filled = euclidean_martix(data, fillna='mean')
```

# Manhattan matrix
Function: manhattan_matrix
### Function takes a pandas DataFrame as input and returns a new DataFrame containing the Manhattan distance between all pairs of columns in the input DataFrame. Manhattan distance is a measure of the distance between two points in a grid system based on the sum of the absolute differences of their coordinates.
## Parameters
data: a pandas DataFrame containing the input data.
fillna (optional): a string indicating the method to use to fill missing values. The options are 'mode', 'median', or 'mean'. If this parameter is not provided or set to None, then the function will not fill missing values in the DataFrame.
## Returns
pandas.DataFrame: A pandas DataFrame containing the Manhattan distance between all pairs of columns in the input DataFrame.

## Example Usage
```python
import pandas as pd
from neulab.Dataframe.discover import manhattan_matrix

# create a sample DataFrame with missing values
data = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [1, 2, 3, None, 5],
    'C': [1, 2, 3, 4, 5]
})

# compute the Manhattan distance matrix with missing values filled with the column median
result_filled = manhattan_matrix(data, fillna='median')
```

# Max matrix
### Function calculates the Max distance between all pairs of columns in a given DataFrame and returns a new DataFrame containing these distances. The Max distance between two vectors is the maximum absolute difference between their corresponding elements.
Function: max_matrix
## Parameters
data: a pandas DataFrame containing the input data.
fillna (optional): a string indicating the method to use to fill missing values. The options are 'mode', 'median', or 'mean'. If this parameter is not provided or set to None, then the function will not fill missing values in the DataFrame.
## Returns
pandas.DataFrame: A DataFrame containing the Max distance between all pairs of columns in the input DataFrame.
## Example Usage
```python
import pandas as pd
from neulab.Dataframe.discover import max_matrix

# create a sample DataFrame with missing values
data = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [1, 2, 3, None, 5],
    'C': [1, 2, 3, 4, 5]
})

# compute the Max distance matrix with missing values filled with the column mode
result_filled = max_matrix(data, fillna='mode')
```

# Correlation matrix
Function: correlation_matrix
### Function takes a Pandas DataFrame as an input and returns a DataFrame containing the correlation matrix between all pairs of columns in the input DataFrame. It computes the correlation coefficients using the method specified in the method parameter.
## Parameters:
data: a pandas DataFrame containing the input data.
method : str (optional, default='pearson'): The method used to compute the correlation coefficients. Default is 'pearson'. Other options are 'kendall' and 'spearman'.
plot : bool (optional, default=False): If True, displays the correlation matrix as a heatmap using the background_gradient function from Pandas Styling.
## Returns:
pandas.DataFrame: A DataFrame containing the correlation matrix between all pairs of columns in the input DataFrame.
## Example Usage
```python
import pandas as pd
from neulab.Dataframe.discover import correlation_matrix

# create a sample DataFrame with missing values
data = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [1, 2, 3, None, 5],
    'C': [1, 2, 3, 4, 5]
})

# compute correlation coefficients
corr_matrix = correlation_matrix(data, method='pearson', plot=True)
```

# Check missing data
Function: check_missing
### Function is used to check for missing values in a given DataFrame and returns a DataFrame containing the number and percentage of missing values for each column in the input DataFrame that has missing values. Additionally, if the plot parameter is set to True, a bar plot of the number of missing values for each feature is displayed.
## Parameters:
data: a pandas DataFrame containing the input data.
plot: bool, optional - If True, a bar plot of the number of missing values for each feature is displayed. Default is False.
## Returns:
pandas.DataFrame - A DataFrame containing the number and percentage of missing values for each column in the input DataFrame that has missing values.
## Example usage:
```python
import pandas as pd
from neulab.Dataframe.discover import check_missing

# create a sample DataFrame with missing values
data = pd.DataFrame({
    'A': [1, None, 3, None, 5],
    'B': [1, 2, 3, None, 5],
    'C': [1, 2, None, 4, 5]
})

# Get misding info
missing_data = check_missing(data, plot=True)
```

# Missing dispersion
Function: plot_missing_value_dispersion
### Function creates a heatmap that shows the locations of missing values in a given DataFrame.
## Parameters
data: a pandas DataFrame containing the input data.
## Returns
None: This function only displays a heatmap and does not return anything.
## Example usage:
```python
import pandas as pd
from neulab.Dataframe.discover import plot_missing_value_dispersion

# create a sample DataFrame with missing values
data = pd.DataFrame({
    'A': [1, None, 3, None, 5],
    'B': [1, 2, 3, None, 5],
    'C': [1, 2, None, 4, 5]
})

# Get misding info
plot_missing_value_dispersion(data)
```

# Plot number of unique values
Function: plot_categorical
### Function takes a Pandas dataframe and creates a bar plot showing the number of unique values for each categorical feature in the dataframe.
## Parameters:
data: Pandas DataFrame containing categorical features to plot.
## Returns:
None: This function only displays a bar plot and does not return anything.
## Example usage:
```python
import pandas as pd
from neulab.Dataframe.discover import plot_categorical

# create a sample DataFrame
data = pd.DataFrame({
    'A': [1, 1, 2, 2, 1],
    'B': [1, 3, 6, 2, 9],
    'C': [4, 2, 7, 4, 5]
})

# Get info
plot_categorical(data)
```

# Scatter plots
Function: plot_scatter
### Function takes a Pandas DataFrame or Series as input and generates scatter plots for each column in the DataFrame. It then displays the plot(s) and does not return anything.
## Parameters:
data: Pandas DataFrame containing categorical features to plot.
## Returns:
None: This function only displays a bar plot and does not return anything.
## Example usage:
```python
import pandas as pd
from neulab.Dataframe.discover import plot_scatter

# create a sample DataFrame
data = pd.DataFrame({
    'A': [1, 1, 2, 2, 1],
    'B': [1, 3, 6, 2, 9],
    'C': [4, 2, 7, 4, 5]
})

# Get info of whole dataframe
plot_scatter(data)

# Get info of only one column of dataframe
plot_scatter(data['A'])
```

# Get categorical columns
Function: get_categorical_columns
### Function takes a Pandas DataFrame as input and detects categorical columns based on a user-defined threshold. The function then returns a list of the column names that are likely to be categorical.
## Parameters:
data: Pandas DataFrame containing categorical features to plot.
threshold: An optional parameter that sets the proportion of unique values below which a column is considered categorical. The default value is 0.05.
## Returns:
list: A list of column names that are likely to be categorical.
## Example usage:
```python
import random
import pandas as pd
from neulab.Dataframe.discover import get_categorical_columns

df = pd.DataFrame({
    'col1': [random.randint(0, 99) for _ in range(50)],
    'col2': [random.randint(0, 100) for _ in range(50)],
    'col3': [random.randint(50, 90) for _ in range(50)],
    'col4': [random.randint(0, 1) for _ in range(50)]
})

categorical = get_categorical_columns(df)

Output:
['col4']
```