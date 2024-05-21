import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
import math
import wandb

"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.19.3
"""


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

