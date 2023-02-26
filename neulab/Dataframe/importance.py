def random_forest_regressor_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Trains a random forest regressor model on the given training data.

    Parameters:
    -----------
    X_train : pandas.DataFrame
        The feature matrix of shape (n_samples, n_features) containing the training input data.
    y_train : pandas.Series
        The target variable of shape (n_samples,) containing the training output data.
    n_estimators : int, optional (default=100)
        The number of trees in the random forest.
    random_state : int or RandomState instance, optional (default=42)
        Controls both the randomness of the bootstrapping of the samples used when building trees and the
        sampling of the features to consider at each split. Pass an int for reproducible results across multiple
        function calls.

    Returns:
    --------
    RandomForestRegressor instance: The trained random forest regressor model.
    """

    from sklearn.ensemble import RandomForestRegressor

     # Train your random forest model
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)

    return rf


def linear_regression_model(X_train, y_train):
    """
    Trains a linear regression model on the given training data.

    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.ndarray
        The feature matrix for training the model.
    y_train : pandas.Series or numpy.ndarray
        The target vector for training the model.

    Returns:
    --------
    sklearn.linear_model.LinearRegression: The trained linear regression model.
    """

    from sklearn.linear_model import LinearRegression

    # Train your linear regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    return lr


def logistic_regression_model(X_train, y_train):
    """
    Trains a logistic regression model using the input training data.

    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.ndarray
        The feature matrix for training the model.
    y_train : pandas.Series or numpy.ndarray
        The target vector for training the model.

    Returns:
    --------
    sklearn.linear_model.LogisticRegression: The trained logistic regression model object
    """

    from sklearn.linear_model import LogisticRegression

    # Train your logistic regression model
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    return lr


def decision_tree_classifier_model(X_train, y_train):
    """
    Trains a decision tree classifier model using the input training data.

    Parameters:
    X_train : pandas.DataFrame or numpy.ndarray
    The feature matrix for training the model.
    y_train : pandas.Series or numpy.ndarray
    The target vector for training the model.

    Returns:
    sklearn.tree.DecisionTreeClassifier: The trained decision tree classifier model object
    """

    from sklearn.tree import DecisionTreeClassifier

    # Train your decision tree classifier model
    dtc = DecisionTreeClassifier(random_state=42)
    dtc.fit(X_train, y_train)

    return dtc


def decision_tree_regressor_model(X_train, y_train, random_state=42):
    """
    Trains a decision tree regressor model using the input training data.

    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.ndarray
        The feature matrix for training the model.
    y_train : pandas.Series or numpy.ndarray
        The target vector for training the model.
    random_state : int, default=42
        Controls the randomness of the estimator. Pass an int for reproducible results.

    Returns:
    --------
    sklearn.tree.DecisionTreeRegressor: The trained decision tree regressor model object
    """

    from sklearn.tree import DecisionTreeRegressor
    
    # Train your decision tree model
    dt = DecisionTreeRegressor(random_state=random_state)
    dt.fit(X_train, y_train)
    
    return dt


def gradient_boosting_regressor_model(X_train, y_train, n_estimators=100, learning_rate=0.1, random_state=42):
    """
    Trains a gradient boosting regression model using the input training data.

    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.ndarray
        The feature matrix for training the model.
    y_train : pandas.Series or numpy.ndarray
        The target vector for training the model.
    n_estimators : int, optional (default=100)
        The number of boosting stages to perform. 
    learning_rate : float, optional (default=0.1)
        The learning rate used in the boosting process.
    random_state : int or RandomState, optional (default=42)
        Controls the random seed used for the pseudo-random number generator.

    Returns:
    --------
    sklearn.ensemble.GradientBoostingRegressor: The trained gradient boosting regression model object
    """
    
    from sklearn.ensemble import GradientBoostingRegressor
    
    # Train your gradient boosting model
    gb = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    gb.fit(X_train, y_train)
    
    return gb


def get_feature_importance(data, target_column, model, plot=False, figsize=(8, 8)):
    """
    Trains the specified model using the input training data and returns a dataframe containing feature importances. If plot=True, a bar chart of feature importances is also displayed.

    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset containing the features and target variable.
    target_column : str
        The name of the column containing the target variable.
    model : str
        The name of the model to train. Must be one of:
            - 'random_forest_regressor'
            - 'linear_regression'
            - 'logistic_regression'
            - 'decision_tree_classifier'
            - 'decision_tree_regressor'
            - 'gradient_boosting_regressor'
    plot : bool, optional
        Whether to display a bar chart of feature importances (default False).
    figsize : tuple, optional
        The size of the figure in inches (default (8, 8)).

    Returns:
    --------
    pandas.DataFrame : A dataframe containing feature importances for the trained model.
    """

    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    df = data.copy()

    # Split your data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df.drop(target_column, axis=1), df[target_column].astype(int), test_size=0.2, random_state=42)

    if model == 'random_forest_regressor':
        pred = random_forest_regressor_model(X_train, y_train)
        # Get feature importance from the trained model
        feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': pred.feature_importances_})
        feature_importances = feature_importances.sort_values('importance', ascending=False)
        model_name = 'Random Forest Regressor'

    elif model == 'linear_regression':
        pred = linear_regression_model(X_train, y_train)
        # Get feature importance from the trained model
        feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': abs(pred.coef_)})
        feature_importances = feature_importances.sort_values('importance', ascending=False)
        model_name = 'Linear Regression'

    elif model == 'logistic_regression':
        pred = logistic_regression_model(X_train, y_train)
        # Get feature importance from the trained model
        feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': abs(pred.coef_[0])})
        feature_importances = feature_importances.sort_values('importance', ascending=False)
        model_name = 'Logistic Regression'

    elif model == 'decision_tree_classifier':
        pred = decision_tree_classifier_model(X_train, y_train)
        # Get feature importance from the trained model
        feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': pred.feature_importances_})
        feature_importances = feature_importances.sort_values('importance', ascending=False)
        model_name = 'Decision Tree Classifier'

    elif model == 'decision_tree_regressor':
        pred = decision_tree_regressor_model(X_train, y_train)
        # Get feature importance from the trained model
        feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': pred.feature_importances_})
        feature_importances = feature_importances.sort_values('importance', ascending=False)
        model_name = 'Decision Tree Regressor'
        
    elif model == 'gradient_boosting_regressor':
        pred = gradient_boosting_regressor_model(X_train, y_train)
        # Get feature importance from the trained model
        feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': pred.feature_importances_})
        feature_importances = feature_importances.sort_values('importance', ascending=False)
        model_name = 'Gradient Boosting Regressor'

    else:
        raise ValueError('Invalid model specified.')
        

    if plot:
        # Define the color map for the bar chart
        colors = plt.cm.get_cmap('RdYlGn')(np.linspace(1, 0, len(feature_importances)))
        
        # Plot the feature importances
        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(feature_importances))
        ax.barh(y_pos, feature_importances['importance'], align='center', color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_importances['feature'], fontsize=10)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'{model_name} Feature Importance', fontsize=14)

        plt.tight_layout()
        plt.show()

    return feature_importances