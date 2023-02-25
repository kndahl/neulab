# Min-Max normalization
### Function is designed to perform Min-Max normalization on selected columns of a given pandas DataFrame.
## Parameters
data: pandas.DataFrame - the DataFrame that needs to be normalized
cols_to_normalize: list - list of column names that needs to be normalized.
## Returns
pandas.DataFrame: Normalized pandas DataFrame.
## Example Usage
```python
import pandas as pd
import numpy as np
from neulab.Dataframe.normalization import min_max_normalize

# Create a sample DataFrame with NaN values
df = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [10, 20, np.nan, 40], 'col3': [100, 200, 300, np.nan]})

# Normalize
df_normalized = min_max_normalize(df, cols_to_normalize=df.columns)

Output:
       col1      col2  col3
0  0.000000  0.000000   0.0
1  0.333333  0.333333   0.5
2  0.666667       NaN   1.0
3  1.000000  1.000000   NaN
```