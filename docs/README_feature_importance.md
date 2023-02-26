# Feature importance
Function: get_feature_importance
### The  function trains a specified model on a provided dataset and returns a dataframe containing feature importances for the trained model. The function also allows for an optional plot of the feature importances in a bar chart.
## Parameters:
data : pandas.DataFrame - The dataset containing the features and target variable.
target_column : str - The name of the column containing the target variable.
model : str - The name of the model to train. Must be one of: 'random_forest_regressor', 'linear_regression', 'logistic_regression', 'decision_tree_classifier', 'decision_tree_regressor', 'gradient_boosting_regressor'.
plot : bool, optional - Whether to display a bar chart of feature importances (default False).
figsize : tuple, optional - The size of the figure in inches (default (8, 8)).
## Returns:
pandas.DataFrame : A dataframe containing feature importances for the trained model.
```python
from sklearn.datasets import fetch_openml
from neulab.Dataframe.normalization import min_max_normalize
from neulab.Dataframe.importance import get_feature_importance

boston = fetch_openml(name='boston')
X = boston.data
y = boston.target
both = X.join(y)
both = min_max_normalize(both, cols_to_normalize=both.drop(columns=['MEDV', 'RAD', 'CHAS']).columns)
importance_df = get_feature_importance(both, target_column='MEDV', model='logistic_regression')

Output:
	feature	importance
0	CRIM	0.635839
5	RM	    0.504893
7	DIS	    0.356908
10	PTRATIO	0.352902
11	B	    0.316413
8	RAD	    0.206935
6	AGE	    0.206316
12	LSTAT	0.163176
3	CHAS	0.155025
2	INDUS	0.154821
1	ZN	    0.147038
4	NOX	    0.070303
9	TAX	    0.059232
```