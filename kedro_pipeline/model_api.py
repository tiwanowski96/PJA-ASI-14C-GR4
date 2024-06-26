from typing import Annotated
from fastapi import FastAPI, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import math
import numpy as np

file_path = "champion.pickle"
app = FastAPI()


class Crab(BaseModel):
    gender: str
    length: float
    diameter: float
    height: float
    weight: float
    shucked_weight: float
    viscara_weight: float
    shell_weight: float

gender_dixt = {
    "M": 2,
    "F": 0,
    "I": 1
}

@app.get("/")
def main():
    return FileResponse(path=file_path, filename=file_path)

@app.post("/model")
async def get_prediction(crab_data: Annotated[Crab, Body(
    examples=[
        {
            "gender": "M",
            "length": 1.5,
            "diameter": 1.2,
            "height": 2.5,
            "weight": 60.6,
            "shucked_weight": 20.6,
            "viscara_weight": 15.8,
            "shell_weight": 15.5,
        }
    ]
    )]
    ):

    volume = crab_data.length * crab_data.height * crab_data.diameter
    weight_proportion = (
        (
            crab_data.shucked_weight +
            crab_data.viscara_weight +
            crab_data.shell_weight
        ) /
        crab_data.weight)
    shucked_proportion = crab_data.shucked_weight / crab_data.weight
    viscera_proportion = crab_data.viscara_weight / crab_data.weight
    shell_proportion = crab_data.shell_weight / crab_data.weight
    shell_area = (crab_data.diameter / 2) ** 2 * math.pi

    scaler = joblib.load('data\\02_intermediate\\scaler.pickle')
    
    scaled_feature = scaler.transform(
        [
            [
                crab_data.length,
                crab_data.diameter,
                crab_data.height,
                crab_data.weight,
                crab_data.shucked_weight,
                crab_data.viscara_weight,
                crab_data.shell_weight,
                volume,
                weight_proportion,
                shell_proportion,
                viscera_proportion,
                shell_proportion,
                shell_area
            ]
        ]
    )
    gender = 1
    input_data = np.insert(scaled_feature, 0, gender)
    model = joblib.load('data\\06_models\\champion\\champion.pickle')
    prediction = model.predict(input_data.reshape(1,-1))
    return prediction.tolist()   

