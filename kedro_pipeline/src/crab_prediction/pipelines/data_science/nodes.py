from typing import Dict, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import logging
import numpy as np
import math
import wandb

logger = logging.getLogger(__name__)

# Input: dataframe, name of target column
# Output: numpy arrays containing splitted data as X_train, X_test, y_train, y_test
def split_data(prepared_dataframe: pd.DataFrame,parameters: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

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

def get_cross_validation_metrics(model: RandomForestRegressor, X_train: np.ndarray, y_train: np.ndarray):

    # Launch 2 simulated experiments
    total_runs = 1
    for run in range(total_runs):
    # 🐝 1️⃣ Start a new run to track this script
        wandb.init(
        # Set the project where this run will be logged
        project="CrabAgePredictionProject", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"experiment_{run+1}", 
        # Track hyperparameters and run metadata
        config={
        "architecture": "RandomForestRegressor",
        "dataset": "CrabAgePrediction.csv",
        "epochs": "5",
        })
    
        # This simple block simulates a training loop logging metrics
        epochs = 5

        for epoch in range(epochs):
            scores = cross_val_score(model, X_train, y_train, cv=5)

            logger = logging.getLogger(__name__)
            logger.info(f"scores: {scores}, mean: {scores.mean()}, std: {scores.std()}")
            # 🐝 2️⃣ Log metrics from your script to W&B
            wandb.log({"cv_scores_mean": scores.mean()})
            
        # Mark the run as finished
        wandb.finish()
    


def get_model_metrics(model: RandomForestRegressor, X_test: np.ndarray, y_test: np.ndarray):

    model_predict = model.predict(X_test)

    mae = mean_absolute_error(y_test, model_predict)
    mse = mean_squared_error(y_test, model_predict)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, model_predict)

    logger = logging.getLogger(__name__)
    logger.info(f"Mae: {mae}\nMse: {mse} \nRMSE: {rmse} \nR2: {r2}")



