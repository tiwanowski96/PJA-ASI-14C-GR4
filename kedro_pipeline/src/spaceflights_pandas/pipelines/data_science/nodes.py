import logging
from typing import Dict, Tuple
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Input: dataframe, name of target column
# Output: numpy arrays containing splitted data as X_train, X_test, y_train, y_test
def split_data(prepared_dataframe: pd.DataFrame,parameters: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    feature = prepared_dataframe.drop(columns=['Age'])
    target = prepared_dataframe['Age']

    X_train, X_test, y_train, y_test =  train_test_split(feature, target, test_size=parameters["test_size"], random_state=parameters["random_state"])

    return X_train, X_test, y_train, y_test


# Input: 2 arrays one containing features and another targets, model max_depth, max_features, n_estimators, min_samples_split, min_samples_leaf
# Notice! Defaults are that of the model provided by sklearn
# Output: Random forest model
def create_rf_model(X_train: np.ndarray, y_train: np.ndarray, 
                    in_max_depth=None, in_max_features=1.0, in_n_estimators=100, in_min_samples_split=2, in_min_samples_leaf=1) -> RandomForestRegressor:

    rf_model = RandomForestRegressor(max_depth=in_max_depth, max_features=in_max_features, n_estimators=in_n_estimators, 
                                     min_samples_leaf=in_min_samples_leaf, min_samples_split=in_min_samples_split)
    rf_model.fit(X_train, y_train)

    return rf_model



