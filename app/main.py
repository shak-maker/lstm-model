from fastapi import FastAPI
from app.schemas.prediction import PredictionInput
from app.services.predictor import predict_glucose

app = FastAPI()

@app.post("/predict")
def predict(data: PredictionInput):
    return predict_glucose(data.values)

@app.get("/")
def root():
    return {"message": "Glucose prediction API is running."}
