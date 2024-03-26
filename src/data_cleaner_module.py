import import_data_module

data = import_data_module.import_dataframe_from_csv("model_data/CrabAgePrediction.csv")

print(data.head())