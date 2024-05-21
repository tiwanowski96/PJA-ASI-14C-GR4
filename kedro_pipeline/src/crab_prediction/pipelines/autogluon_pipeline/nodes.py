from autogluon.tabular import TabularDataset, TabularPredictor
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Input: dataframe
# Output: numpy arrays containing splitted data as train, test
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

def evaluate_autogluon_model(predicator: TabularPredictor, test_set: pd.DataFrame):

    leaderboards = predicator.leaderboard(test_set)
    metrics_dic = predicator.evaluate(test_set, silent=True)

    logger.info(
        f'Metrics: \n {metrics_dic}'
    )