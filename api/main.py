# api/main.py
from fastapi import FastAPI, Request
from api.models import CropFeatures, RecommendationRequest
import joblib
import joblib
import os
import pickle
import pandas as pd
from pydantic import BaseModel
from rdflib import Graph, Namespace, RDF, RDFS
import difflib
import uvicorn

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

# Load TTL ontology from GitHub
TTL_URL = "https://raw.githubusercontent.com/LENISH-V/Smart-Farminng-Bot/main/Smart_Farming.ttl"
g = Graph()
g.parse(TTL_URL, format="ttl")
 
sf = Namespace("http://www.semanticweb.org/lensv/ontologies/2025/2/untitled-ontology-13/")
 
class Query(BaseModel):
    text: str
 
def get_labels_by_class(klass):
    return [
        str(label)
        for s in g.subjects(RDF.type, klass)
        for label in g.objects(s, RDFS.label)
    ]
 
def fuzzy_match(text, choices, cutoff=0.6):
    match = difflib.get_close_matches(text, choices, n=1, cutoff=cutoff)
    return match[0] if match else None
 
@app.post("/match")
async def match_query(query: Query):
    text = query.text.lower()
 
    crops = get_labels_by_class(sf.Crop)
    ferts = get_labels_by_class(sf.Fertilization)
    soils = get_labels_by_class(sf.Soil)
    weathers = get_labels_by_class(sf.Weather)
 
    matched_crop = fuzzy_match(text, crops)
    matched_fert = fuzzy_match(text, ferts)
    matched_soil = fuzzy_match(text, soils)
    matched_weather = fuzzy_match(text, weathers)
 
    linked_fert = None
    if matched_crop:
        for crop in g.subjects(RDF.type, sf.Crop):
            label = g.value(crop, RDFS.label)
            if label and matched_crop.lower() == str(label).lower():
                fert_uri = g.value(crop, sf.receivesFertilization)
                if fert_uri:
                    fert_label = g.value(fert_uri, RDFS.label)
                    if fert_label:
                        linked_fert = str(fert_label)
                break
 
    rag_parts = []
    if matched_crop: rag_parts.append(f"crop: {matched_crop}")
    if linked_fert or matched_fert: rag_parts.append(f"fertilizer: {linked_fert or matched_fert}")
    if matched_soil: rag_parts.append(f"soil: {matched_soil}")
    if matched_weather: rag_parts.append(f"weather: {matched_weather}")
    rag_query = ", ".join(rag_parts)
 
    return {
        "ontology_match": bool(matched_crop or matched_fert or linked_fert),
        "matched_crop": matched_crop,
        "matched_fertilizer": linked_fert or matched_fert,
        "matched_soil": matched_soil,
        "matched_weather": matched_weather,
        "rag_query": rag_query,
        "folder_name": "RAG_files",
        "raw_user_text": text
    }
