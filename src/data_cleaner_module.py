import pandas as pd

def clean_Data(dataframe) -> pd.DataFrame:

    dataframe = dataframe[dataframe['Viscera Weight'] < 17.5]
    dataframe = dataframe[dataframe['Shell Weight'] <= 24]
    dataframe = dataframe[dataframe['Shucked Weight'] <= 36]
    dataframe = dataframe[dataframe['Weight'] <= 78]
    dataframe = dataframe[dataframe['Height'] < 1]

    return dataframe