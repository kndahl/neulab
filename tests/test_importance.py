import pandas as pd
from sklearn.datasets import fetch_openml
from neulab.Dataframe.importance import random_forest_regressor_model
from neulab.Dataframe.importance import random_forest_regressor_model
from neulab.Dataframe.importance import linear_regression_model
from neulab.Dataframe.importance import logistic_regression_model
from neulab.Dataframe.importance import decision_tree_classifier_model
from neulab.Dataframe.importance import decision_tree_regressor_model
from neulab.Dataframe.importance import gradient_boosting_regressor_model
from neulab.Dataframe.importance import get_feature_importance
from neulab.Dataframe.normalization import min_max_normalize

def test_random_forest_regressor_model():
    boston = fetch_openml(name='boston')
    X = boston.data
    y = boston.target
    model = random_forest_regressor_model(X, y.astype(int))
    assert hasattr(model, 'predict')
    
def test_linear_regression_model():
    boston = fetch_openml(name='boston')
    X = boston.data
    y = boston.target
    model = linear_regression_model(X, y.astype(int))
    assert hasattr(model, 'predict')
    
def test_logistic_regression_model():
    boston = fetch_openml(name='boston')
    X = boston.data
    y = boston.target
    model = logistic_regression_model(X, y.astype(int))
    assert hasattr(model, 'predict')
    
def test_decision_tree_classifier_model():
    boston = fetch_openml(name='boston')
    X = boston.data
    y = boston.target
    model = decision_tree_classifier_model(X, y.astype(int))
    assert hasattr(model, 'predict')
    
def test_decision_tree_regressor_model():
    boston = fetch_openml(name='boston')
    X = boston.data
    y = boston.target
    model = decision_tree_regressor_model(X, y.astype(int))
    assert hasattr(model, 'predict')
    
def test_gradient_boosting_regressor_model():
    boston = fetch_openml(name='boston')
    X = boston.data
    y = boston.target
    model = gradient_boosting_regressor_model(X, y.astype(int))
    assert hasattr(model, 'predict')
    
def test_get_feature_importance():    
    boston = fetch_openml(name='boston')
    X = boston.data
    y = boston.target
    both = X.join(y)
    both = min_max_normalize(both, cols_to_normalize=both.drop(columns=['MEDV', 'RAD', 'CHAS']).columns)
    importance_df = get_feature_importance(both, target_column='MEDV', model='random_forest_regressor')
    assert isinstance(importance_df, pd.DataFrame)
    assert all(col in importance_df.columns for col in ['feature', 'importance'])