from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# load scaler
with open("scaler_weights.pkl", "rb") as f:
    scaler = pickle.load(f)

# load model
with open("model_weights.pkl", "rb") as f:
    model = pickle.load(f)


class InputData(BaseModel):
    features: list[float]


@app.get("/")
def home():
    return {"message": "Breast Cancer Classification API"}


@app.post("/predict")
def predict(data: InputData):

    features = np.array(data.features).reshape(1, -1)

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    result = int(prediction[0][0] > 0.5)

    return {"prediction": result}