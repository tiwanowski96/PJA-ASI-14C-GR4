import pandas as pd

# Take path to csv file and return pandas Dataframe. If file not found return empty dataframe. If error return null.
def import_dataframe_from_csv(path) -> pd.DataFrame:
    try:
        imported_data = pd.read_csv(path)
        return imported_data
    
    except FileNotFoundError:
        print(f"Error: File not found in '{path}'")
        return pd.DataFrame()
    
    except Exception as exp:
        print(f"Error: {exp}")
        return None
    