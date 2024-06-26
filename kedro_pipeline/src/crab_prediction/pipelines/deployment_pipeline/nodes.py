from __future__ import annotations
from autogluon.tabular import TabularPredictor
from typing import Dict, Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import logging
import os
import pickle

logger = logging.getLogger(__name__)

   
# def compare_models(my_bool:bool,automl_challenger: dict,random_challenger: dict) -> None:
#     champion_accuracy = 0.90#champion['accuracy']
#     if my_bool==True:
#         challenger_accuracy = automl_challenger['accuracy']
#     else:
#         challenger_accuracy = random_challenger['accuracy']

#     if challenger_accuracy > champion_accuracy:
#         logger.info("Challenger model is better than the champion model.")
#     else:
#         logger.info("Challenger model is not better than the champion model.")
#     pass

def load_champion_model() -> Union[TabularPredictor, RandomForestRegressor, str]:
    models_path = "data//06_models//champion"
    champion_model_path = os.path.join(models_path, "champion.pickle")
    if os.path.isfile(champion_model_path):
        with open(champion_model_path, 'rb') as file:
            champion_model = pickle.load(file)
        return champion_model
    else:
        return ""
    


def compare_models(challenger_model:TabularPredictor | RandomForestRegressor, champion_model:TabularPredictor | RandomForestRegressor | str, datasets_list: list, parameters:Dict) -> TabularPredictor | RandomForestRegressor:
  

    if isinstance(champion_model, RandomForestRegressor) and parameters["model_creation"]:

        rf_champion_pred = champion_model.predict(datasets_list[3])
        r2_rf_champion = r2_score(datasets_list[5], rf_champion_pred)

        challenger_model_pred = challenger_model.evaluate(datasets_list[1])
        r2_tabular_challenger = challenger_model_pred.get('r2')

        if r2_rf_champion>r2_tabular_challenger:
            logger.info("Champion model is better than the challenger model.")
            return champion_model
        else:
            logger.info("Challenger model is better than the champion model.")
            return challenger_model

    elif isinstance(champion_model, RandomForestRegressor) and not parameters["model_creation"]:

        
        rf_champion_pred = champion_model.predict(datasets_list[3])
        r2_rf_champion = r2_score(datasets_list[5], rf_champion_pred)

        rf_challenger_pred = challenger_model.predict(datasets_list[3])
        r2_rf_challenger = r2_score(datasets_list[5], rf_challenger_pred)
        logger.info(f"r2_rf_challenger: {r2_rf_challenger}, r2_rf_champion: {r2_rf_champion}")

        if r2_rf_champion > r2_rf_challenger:
            logger.info("Champion model is better than the challenger model.")
            return champion_model
        else:
            logger.info("Challenger model is better than the champion model.")
            return challenger_model

    elif isinstance(champion_model, TabularPredictor) and parameters["model_creation"]:

        tabular_champion_pred = champion_model.evaluate(datasets_list[1])
        r2_tabular_champion = tabular_champion_pred.get('r2')


        challenger_model_pred = challenger_model.evaluate(datasets_list[1])
        r2_tabular_challenger = challenger_model_pred.get('r2')

        if r2_tabular_champion > r2_tabular_challenger:
            logger.info("Champion model is better than the challenger model.")
            return champion_model
        else:
            logger.info("Challenger model is better than the champion model.")
            return challenger_model

    elif isinstance(champion_model, TabularPredictor) and not parameters["model_creation"]:
        tabular_champion_pred = champion_model.evaluate(datasets_list[1])
        r2_tabular_champion = tabular_champion_pred.get('r2')

        rf_challenger_pred = challenger_model.predict(datasets_list[3])
        r2_rf_challenger = r2_score(datasets_list[5], rf_challenger_pred)

        logger.info(f"r2_rf_challenger: {r2_rf_challenger}, r2_rf_champion: {r2_tabular_champion}")
        if r2_tabular_champion > r2_rf_challenger:
            logger.info("Champion model is better than the challenger model.")
            return champion_model
        else:
            logger.info("Challenger model is better than the champion model.")
            return challenger_model
    else:
        logger.info("Champion model does not exists, challenger model is new champion.")
        return challenger_model