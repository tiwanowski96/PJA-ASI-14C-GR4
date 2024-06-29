from autogluon.tabular import TabularDataset, TabularPredictor
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def split_data_for_autogluon(preprocessed_data: pd.DataFrame, parameters: Dict) -> Tuple[np.ndarray, np.ndarray]:
    logger.info(
        f'Split used parameters:\n \
        test_size: {parameters["test_size"]}\n \
        random_state: {parameters["random_state"]}'
    )

    train = preprocessed_data.sample(frac=(1-parameters["test_size"]), random_state=parameters["random_state"])
    test = preprocessed_data.drop(train.index)

    return train, test

def create_autogluon_model(train_set: pd.DataFrame, parameters: Dict) -> TabularPredictor:

    logger.info(
        f'Autogluon used parameters:\n \
        target_column: {parameters["target_column"]}\n \
        path: {parameters["path"]}\n \
        problem_type: {parameters["problem_type"]}\n \
        eval_metric: {parameters["eval_metric"]}\n \
        presets: {parameters["presets"]}'
    )

    train_data = TabularDataset(train_set)

    predictor = TabularPredictor(label=parameters["target_column"],path=parameters["path"],problem_type=parameters["problem_type"], eval_metric=parameters["eval_metric"]).fit(train_data, presets=parameters["presets"])

    return predictor


def split_data_for_rf(prepared_dataframe: pd.DataFrame,parameters: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    logger.info(
        f'Used parameters:\n \
        test_size: {parameters["test_size"]}\n \
        random_state: {parameters["random_state"]}'
    )

    feature = prepared_dataframe.drop(columns=['Age'])
    target = prepared_dataframe['Age']

    X_train, X_test, y_train, y_test =  train_test_split(feature, target, test_size=parameters["test_size"], random_state=parameters["random_state"])

    return X_train, X_test, y_train, y_test


# Input: 2 arrays one containing features and another targets, model max_depth, max_features, n_estimators, min_samples_split, min_samples_leaf
# Notice! Defaults are that of the model provided by sklearn
# Output: Random forest model
def create_rf_model(X_train: np.ndarray, y_train: np.ndarray, parameters: Dict) -> RandomForestRegressor:

    logger.info(
        f'Used parameters:\n \
        max_depth: {parameters["max_depth"]}\n \
        max_features: {parameters["max_features"]}\n \
        n_estimators: {parameters["n_estimators"]}\n \
        min_samples_leaf: {parameters["min_samples_leaf"]}\n \
        min_samples_split: {parameters["min_samples_split"]}\n' 
    )

    rf_model = RandomForestRegressor(
        max_depth=parameters["max_depth"],
        max_features=parameters["max_features"],
        n_estimators=parameters["n_estimators"], 
        min_samples_leaf=parameters["min_samples_leaf"],
        min_samples_split=parameters["min_samples_split"]
    )
    rf_model.fit(X_train, y_train)

    return rf_model
    
def split_data(prepared_dataframe: pd.DataFrame,parameters: Dict) -> list:
    train, test = split_data_for_autogluon(prepared_dataframe, parameters)
    X_train, X_test, y_train, y_test = split_data_for_rf(prepared_dataframe, parameters)
    datasets_list = [train, test, X_train, X_test, y_train, y_test]
    return datasets_list

def create_model(datasets_list: list, parameters: Dict) -> TabularPredictor | RandomForestRegressor:
    if parameters["model_creation"]:
        return create_autogluon_model(datasets_list[0], parameters)
    else:
        return create_rf_model(datasets_list[2], datasets_list[4], parameters)
