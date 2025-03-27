from fastapi import FastAPI
from model import CropFeatures
import joblib
import numpy as np
import os

# Create FastAPI instance
app = FastAPI(
    title="Smart Farming Crop Recommendation API",
    description="Predict the most suitable crop based on soil and environmental conditions 🌱",
    version="1.0.0"
)

# Load trained model from the correct folder
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml_model", "crop_model.pkl"))

try:
    model = joblib.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the Smart Farming Recommender API!"}

# Predict endpoint
@app.post("/predict")
def predict_crop(data: CropFeatures):
    try:
        if model is None:
            return {"error": "Model not loaded"}

        features = np.array([[
            data.N,
            data.P,
            data.K,
            data.temperature,
            data.humidity,
            data.ph,
            data.rainfall
        ]])
        prediction = model.predict(features)
        return {"recommended_crop": str(prediction[0])}
    except Exception as e:
        return {"error": str(e)}
