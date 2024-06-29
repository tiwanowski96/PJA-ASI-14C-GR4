import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


def clean_Data(dataframe: pd.DataFrame) -> pd.DataFrame:

    if len(dataframe) == 0:
        print("Error: The data is empty.")
        return None

    dataframe = dataframe[dataframe['Viscera Weight'] < 17.5]
    dataframe = dataframe[dataframe['Shell Weight'] <= 24]
    dataframe = dataframe[dataframe['Shucked Weight'] <= 36]
    dataframe = dataframe[dataframe['Weight'] <= 78]
    dataframe = dataframe[dataframe['Height'] < 1]
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe


def prepare_cleaned_data(data_cleaned: pd.DataFrame):

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
    

    return prepared_dataframe, scaler


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


