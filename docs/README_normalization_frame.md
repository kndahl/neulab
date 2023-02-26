# Min-Max normalization
Function: min_max_normalize
### Function is designed to perform Min-Max normalization on selected columns of a given pandas DataFrame.
This method scales the data to a fixed range between 0 and 1. It is calculated as (x - min)/(max - min), where x is the original value, min is the minimum value of the feature, and max is the maximum value of the feature.
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
This method standardizes the data to have zero mean and unit variance. It is calculated as (x - mean)/standard deviation, where x is the original value, mean is the mean of the feature, and standard deviation is the standard deviation of the feature.
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
2	0.387298	 NaN            1.0
3	1.161895	1.091089	NaN
```

# Log transformation
Function: log_transform
### Function is designed to perform a log transformation on the selected columns of the given pandas DataFrame.
This method applies a logarithmic transformation to the data, which can be useful for data that has a skewed distribution. It is calculated as log(x), where x is the original value.
## Parameters
data: pandas.DataFrame - the DataFrame that needs to be normalized
cols_to_transform: list - list of column names that needs to be transformed.
## Returns
pandas.DataFrame: Normalized pandas DataFrame.
## Example Usage
```python
import pandas as pd
import numpy as np
from neulab.Dataframe.normalization import log_transform

# Create a sample DataFrame
df = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [10, 20, np.nan, 40], 'col3': [100, 200, 300, np.nan]})

# Transform
df_transform = log_transform(df, cols_to_transform=df.columns)

Output:

              col1	col2	       col3
0	0.000000	2.302585	4.605170
1	0.693147	2.995732	5.298317
2	1.098612	NaN	        5.703782
3	1.386294	3.688879	NaN
```

# Power transformation
Function: power_transform
### Function is designed to perform a power transformation on the selected columns of the given pandas DataFrame.
This method applies a power transformation to the data, which can be useful for data that has a skewed distribution. It is calculated as x^lambda, where x is the original value and lambda is a parameter that can be tuned.
## Parameters
data: pandas.DataFrame - the DataFrame that needs to be normalized
cols_to_transform: list - list of column names that needs to be transformed.
power: float - the power to raise the values to. Default is 1 (no transformation).
## Returns
pandas.DataFrame: Normalized pandas DataFrame.
## Example Usage
```python
import pandas as pd
import numpy as np
from neulab.Dataframe.normalization import power_transform

# Create a sample DataFrame
df = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [10, 20, np.nan, 40], 'col3': [100, 200, 300, np.nan]})

# Transform
df_transform = power_transform(df, df.columns, power=2)

Output:


       col1	col2	       col3
0	1	100.0	       10000.0
1	4	400.0	       40000.0
2	9	NaN	       90000.0
3	16	1600.0	       NaN
```