# Min-Max normalization
Function: min_max_normalize
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

# Create a sample DataFrame
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

# Z-score normalization
Function: z_score_normalize
### Function is designed to perform Z-score normalization on the selected columns of the given pandas DataFrame.
## Parameters
data: pandas.DataFrame - the DataFrame that needs to be normalized
cols_to_normalize: list - list of column names that needs to be normalized.
## Returns
pandas.DataFrame: Normalized pandas DataFrame.
## Example Usage
```python
import pandas as pd
import numpy as np
from neulab.Dataframe.normalization import z_score_normalize

# Create a sample DataFrame
df = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [10, 20, np.nan, 40], 'col3': [100, 200, 300, np.nan]})

# Normalize
df_normalized = z_score_normalize(df, cols_to_normalize=df.columns)

Output:

              col1          col2	col3
0	-1.161895	-0.872872	-1.0
1	-0.387298	-0.218218	0.0
2	0.387298	 NaN          1.0
3	1.161895	1.091089	NaN
```