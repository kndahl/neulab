# Simple imputation
### Function performs simple imputation for NaN values in a pandas DataFrame using the specified method. The imputation methods available are 'mean', 'median', and 'mode'. The function returns a new DataFrame with NaN values imputed.
## Parameters
data : pandas.DataFrame
The input DataFrame containing NaN values to be imputed.
method : str, optional, default='mean'
The method used to impute the missing values. Valid options are 'mean', 'median', 'mode'.
## Returns:
pandas.DataFrame: A DataFrame with NaN values imputed using the specified method.
## Example Usage
```python
import pandas as pd
import numpy as np

from neulab.Dataframe.recover import simple_imputation

data = pd.DataFrame({
        'A': [1, 2, np.nan, np.nan, 5],
        'B': [6, 7, 7, np.nan, 10]
    })

df_recovered = simple_imputation(data, method='mode')

Output:
NaN values imputed with mode.
        A	B
0	1.0	6.0
1	2.0	7.0
2	1.0	7.0
3	1.0	7.0
4	5.0	10.0
```

# Distance imputation
### Function performs imputation for missing values in a Pandas DataFrame using the distance method. It calculates the distance between the missing values and other values in the DataFrame using one of the three distance metrics: 'euclidean', 'manhattan', or 'max'. The function then replaces the missing values with the value that is closest to them, based on the chosen distance metric.
## Parameters:
data: pandas.DataFrame
The input DataFrame containing NaN values to be imputed.
metric: str, optional, default='euclidean'
The metric used to impute the missing values. Valid options are 'euclidean', 'manhattan', or 'max'.
## Returns:
pandas.DataFrame: A DataFrame with NaN values imputed using the specified method.
## Example Usage
```python
import pandas as pd
import numpy as np

from neulab.Dataframe.recover import distance_imputation

data = pd.DataFrame({
        'A': [1, 2, np.nan, np.nan, 5],
        'B': [6, 7, 7, np.nan, 10]
    })

df_recovered = distance_imputation(data, metric='euclidean')

Output:
NaN values imputed with euclidean metric.
        A	B
0	1.0	6.0
1	2.0	7.0
2	1.0	7.0
3	1.0	6.0
4	5.0	10.0
```
# Iterative imputation
### Function imputes missing values in a given DataFrame using the IterativeImputer from scikit-learn. The algorithm works by modeling the missing values as a function of the other features in the dataset.
It starts by filling in missing values with initial estimates, such as the mean or median of the non-missing values. It then uses this initial estimate to fit a model to the non-missing values and predict the missing values. This process is repeated multiple times, with the predicted values from each iteration being used as the input for the next iteration.

IterativeImputer is useful when the missing values in a dataset are not completely at random, and there is some pattern or relationship between the missing values and the other features. However, it is important to note that this method can be computationally expensive, especially for large datasets or when using complex regression models.

## Parameters:
data : pandas.DataFrame
The input DataFrame.
## Returns:
pandas.DataFrame:
A DataFrame with missing values imputed using IterativeImputer, if any. If there are no missing values, the function prints "No need to impute data." and returns the original DataFrame.
```python
import pandas as pd
import numpy as np

from neulab.Dataframe.recover import iterative_imputation

data = pd.DataFrame({
        'A': [1, 2, np.nan, np.nan, 5],
        'B': [6, 7, 7, np.nan, 10]
    })

df_recovered = iterative_imputation(data)

Output:
Imputing NaN using IterativeImputer
        A	B
0	1.0	6.0
1	2.0	7.0
2	2.0	7.0
3	2.5	7.5
4	5.0	10.0
```