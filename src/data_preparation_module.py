import pandas as pd
import numpy as np
import math
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# Input: dataframe with clean data 
# Output: dataframe with standardization and encoded data
def prepare_cleaned_data(data_cleaned: pd.DataFrame) -> pd.DataFrame:

    # standarization part

    scaler = StandardScaler()

    cols_to_scale = data_cleaned.select_dtypes(include=['float', 'int']).drop(columns="Age").columns

    scaled_data = scaler.fit_transform(data_cleaned[cols_to_scale])
    scaled_dataframe = pd.DataFrame(scaled_data, columns=cols_to_scale)
    scaled_dataframe = pd.concat([scaled_dataframe, data_cleaned.drop(columns=cols_to_scale)], axis=1)

    # encoding part

    label_encoder = LabelEncoder()

    cols_to_encode = data_cleaned.select_dtypes(include=['object']).columns

    encoded_data = label_encoder.fit_transform(data_cleaned[cols_to_encode[0]].values)
    encoded_dataframe = pd.DataFrame(encoded_data, columns=cols_to_encode)
    prepared_dataframe = pd.concat([encoded_dataframe, scaled_dataframe.drop(columns=cols_to_encode)], axis=1)

    return prepared_dataframe


# Input: dataframe for random forest
# Output: dataframe with new features
def enrich_rf_features(dataframe: pd.DataFrame) -> pd.DataFrame:

    dataframe["Volume"] = dataframe["Length"] * dataframe["Height"] * dataframe["Diameter"]
    dataframe["Weight proportion"] = (dataframe["Shucked Weight"] + dataframe["Viscera Weight"] + dataframe["Shell Weight"]) / dataframe["Weight"]
    dataframe["Shucked proportion"] = dataframe["Shucked Weight"] / dataframe["Weight"]
    dataframe["Viscera proportion"] = dataframe["Viscera Weight"] / dataframe["Weight"]
    dataframe["Shell proportion"] = dataframe["Shell Weight"] / dataframe["Weight"]
    dataframe["Shell area"] = (dataframe["Diameter"] / 2)**2 * math.pi

    return dataframe

# Input: dataframe for ridge
# Output: dataframe with new features
def enrich_ridge_features(dataframe: pd.DataFrame) -> pd.DataFrame:

    dataframe["Volume"] = dataframe["Length"] * dataframe["Height"] * dataframe["Diameter"]
    dataframe["Weight proportion"] = (dataframe["Shucked Weight"] + dataframe["Viscera Weight"] + dataframe["Shell Weight"]) / dataframe["Weight"]
    dataframe["Shucked proportion"] = dataframe["Shucked Weight"] / dataframe["Weight"]
    dataframe["Viscera proportion"] = dataframe["Viscera Weight"] / dataframe["Weight"]
    dataframe["Shell proportion"] = dataframe["Shell Weight"] / dataframe["Weight"]

    return dataframe

# Input: dataframe, name of target column
# Output: numpy arrays containing splitted data as X_train, X_test, y_train, y_test
def split_data(prepared_dataframe: pd.DataFrame, target_name: str, in_test_size=0.35, in_random_state=0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    feature = prepared_dataframe.drop(columns=[target_name])
    target = prepared_dataframe[target_name]

    X_train, X_test, y_train, y_test =  train_test_split(feature, target, test_size=in_test_size, random_state=in_random_state)

    return X_train, X_test, y_train, y_test
