
import sys
import numpy as np
import wandb

from download.import_data_module import import_dataframe_from_csv 
from data_preprocess.data_cleaner_module import clean_Data 
from data_preprocess.data_preparation_module import prepare_cleaned_data, enrich_rf_features, enrich_ridge_features, split_data
from ml.model_creator_module import create_ridge_model, create_rf_model
from evaluation.model_metrics_module import get_model_metrics, get_cross_validation_metrics, model_params_random_search

if __name__ == '__main__':
    crabs = import_dataframe_from_csv(path="model_data\CrabAgePrediction.csv")
    crabs = clean_Data(crabs)
    crabs.reset_index(drop=True, inplace=True)
    crabs = enrich_rf_features(crabs)
    crabs = prepare_cleaned_data(crabs)
    X_train, X_test, y_train, y_test = split_data(crabs, target_name="Age")
    rf_model = create_rf_model(X_train, y_train, in_max_depth=7, in_max_features='sqrt', in_min_samples_leaf=2, in_min_samples_split=6, in_n_estimators=215)
    model_predict, mae, mse, rmse, r2 = get_model_metrics(rf_model, X_test, y_test)
    print(f"Mae: {mae}\n Mse: {mse} \n RMSE: {rmse} \n R2: {r2}")

    #cv_scores, cv_scores_mean, cv_scores_std = get_cross_validation_metrics(rf_model, X_train, y_train)
    # Launch 5 simulated experiments
    total_runs = 5
    for run in range(total_runs):
    # 🐝 1️⃣ Start a new run to track this script
        wandb.init(
        # Set the project where this run will be logged
        project="CrabAgePredictionProject", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"experiment_{run+1}", 
        # Track hyperparameters and run metadata
        config={
        "architecture": rf_model,
        "dataset": "CrabAgePrediction.csv",
        "epochs": "10",
        })
    
        # This simple block simulates a training loop logging metrics
        epochs = 10

        for epoch in range(epochs):
            cv_scores, cv_scores_mean, cv_scores_std = get_cross_validation_metrics(rf_model, X_train, y_train)
            
            # 🐝 2️⃣ Log metrics from your script to W&B
            wandb.log({"cv_scores_mean": cv_scores_mean})
            
        # Mark the run as finished
        wandb.finish()
    print(f"Scores: {cv_scores}\n Mean: {cv_scores_mean}\n Devation: {cv_scores_std}")
    