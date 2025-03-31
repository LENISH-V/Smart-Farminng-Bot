# api/main.py
from fastapi import FastAPI
from api.models import CropFeatures, RecommendationRequest
import joblib
import joblib
import os
import pickle
import pandas as pd
from pydantic import BaseModel

# Create FastAPI instance
app = FastAPI(
    title="Smart Farming Crop Recommendation API",
    description="Predict the most suitable crop based on soil and weather data",
    version="2.0"
)

# Load the trained model
model_path = "ml_model/crop_model.pkl"
model = joblib.load(model_path)


with open("/workspaces/Smart-Farminng-Bot/ml_model/best_fert.pkl", "rb") as f:
    best_model = pickle.load(f)

# Define a mapping for fertilizer names to their NPK values.
npk_mapping = {
    "Urea": {"Nitrogen": 37, "Phosphorous": 0, "Potassium": 0},
    "DAP": {"Nitrogen": 12, "Phosphorous": 36, "Potassium": 0},
    "14-35-14": {"Nitrogen": 7, "Phosphorous": 30, "Potassium": 9},
    "28-28": {"Nitrogen": 22, "Phosphorous": 20, "Potassium": 0}
}


def recommend_fertilizer(input_data: dict):
    """
    Given input_data, predicts the fertilizer recommendation and returns 
    the fertilizer name along with its corresponding NPK values.
    
    Expected keys in input_data:
        'Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type'
    """
    # Convert the input dictionary into a DataFrame.
    input_df = pd.DataFrame([input_data])
    
    # Predict the fertilizer recommendation using the loaded model.
    fertilizer_pred = best_model.predict(input_df)[0]
    
    # Retrieve the NPK values for the predicted fertilizer.
    npk_values = npk_mapping.get(fertilizer_pred, {})
    
    return fertilizer_pred, npk_values

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
    
DEFAULT_TEMPERATURE = 30.0
DEFAULT_HUMIDITY = 60.0
DEFAULT_MOISTURE = 40.0

@app.post("/recommend")
def recommend(req: RecommendationRequest):
    # Construct the input dictionary with defaults and user inputs
    input_data = {
        "Temparature": DEFAULT_TEMPERATURE,  # Match spelling as in training data
        "Humidity": DEFAULT_HUMIDITY,
        "Moisture": DEFAULT_MOISTURE,
        "Soil Type": req.soil_type,
        "Crop Type": req.crop_type
    }
    
    fertilizer, npk = recommend_fertilizer(input_data)
    return {"fertilizer": fertilizer, "npk": npk}
