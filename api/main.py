# api/main.py
from fastapi import FastAPI
from api.models import CropFeatures
import joblib
import numpy as np
import joblib
import os

# Create FastAPI instance
app = FastAPI(
    title="Smart Farming Crop Recommendation API",
    description="Predict the most suitable crop based on soil and weather data",
    version="2.0"
)

# Load the trained model
model_path = "ml_model/crop_model.pkl"
model = joblib.load(model_path)

# Default values for missing inputs
DEFAULTS = {
    "N": 50,
    "P": 40,
    "K": 40,
    "temperature": 25.0,
    "humidity": 70.0,
    "ph": 6.5,
    "rainfall": 120.0
}

@app.get("/")
def read_root():
    return {"message": "Welcome to Smart Farming Crop Recommender!"}

@app.post("/predict_crop")
def predict_crop(data: CropFeatures):
    try:
        # Use user input or fallback to default
        features = [
            data.N if data.N is not None else DEFAULTS["N"],
            data.P if data.P is not None else DEFAULTS["P"],
            data.K if data.K is not None else DEFAULTS["K"],
            data.temperature if data.temperature is not None else DEFAULTS["temperature"],
            data.humidity if data.humidity is not None else DEFAULTS["humidity"],
            data.ph if data.ph is not None else DEFAULTS["ph"],
            data.rainfall if data.rainfall is not None else DEFAULTS["rainfall"],
        ]
        prediction = model.predict([features])
        return {"recommended_crop": prediction[0]}
    except Exception as e:
        return {"error": str(e)}
