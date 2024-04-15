import logging
from typing import Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
import math

"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.19.3
"""


def get_cross_validation_metrics(model: RandomForestRegressor, X_train: np.ndarray, y_train: np.ndarray):

    scores = cross_val_score(model, X_train, y_train, cv=5)


    logger = logging.getLogger(__name__)
    logger.info(f"scores: {scores}, mean: {scores.mean()}, std: {scores.std()}")



def get_model_metrics(model: RandomForestRegressor, X_test: np.ndarray, y_test: np.ndarray):

    model_predict = model.predict(X_test)

    mae = mean_absolute_error(y_test, model_predict)
    mse = mean_squared_error(y_test, model_predict)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, model_predict)

    logger = logging.getLogger(__name__)
    logger.info(f"Mae: {mae}\n Mse: {mse} \n RMSE: {rmse} \n R2: {r2}")

