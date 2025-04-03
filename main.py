from fastapi import FastAPI, Query
from rdflib import Graph, Namespace, RDF, RDFS, OWL
from typing import List, Optional
import uvicorn

app = FastAPI()

# Load TTL file
g = Graph()
g.parse("smart_farming_enriched_labeled.ttl", format="ttl")

sf = Namespace("http://www.smartfarming.org/ontology#")

def get_labels_for_class(klass_uri):
    return [
        str(label)
        for s in g.subjects(RDF.type, klass_uri)
        for label in g.objects(s, RDFS.label)
    ]

def get_values_by_object_property(property_uri, value_uri):
    return [
        str(subject)
        for subject, obj in g.subject_objects(property_uri)
        if obj == value_uri
    ]

def get_objects_for_subject(subject_uri, property_uri):
    return [
        str(obj)
        for obj in g.objects(subject_uri, property_uri)
    ]

def find_uri_by_label(label):
    for s, p, o in g.triples((None, RDFS.label, None)):
        if str(o).lower() == label.lower():
            return s
    return None

@app.get("/crops")
def get_all_crops():
    return get_labels_for_class(sf.Crop)

@app.get("/crops/by-soil")
def get_crops_by_soil(soil: str):
    soil_uri = find_uri_by_label(soil)
    if not soil_uri:
        return {"error": "Soil not found"}
    return get_values_by_object_property(sf.grownIn, soil_uri)

@app.get("/fertilizer/by-crop")
def get_fertilizer_by_crop(crop: str):
    crop_uri = find_uri_by_label(crop)
    if not crop_uri:
        return {"error": "Crop not found"}
    return get_objects_for_subject(crop_uri, sf.receivesFertilizer)

@app.get("/diseases/by-crop")
def get_diseases_by_crop(crop: str):
    crop_uri = find_uri_by_label(crop)
    if not crop_uri:
        return {"error": "Crop not found"}
    return get_objects_for_subject(crop_uri, sf.hasDisease)

@app.get("/symptoms/by-disease")
def get_symptoms_by_disease(disease: str):
    disease_uri = find_uri_by_label(disease)
    if not disease_uri:
        return {"error": "Disease not found"}
    return get_objects_for_subject(disease_uri, sf.showsSymptom)

@app.get("/insecticides/by-disease")
def get_insecticide_by_disease(disease: str):
    disease_uri = find_uri_by_label(disease)
    if not disease_uri:
        return {"error": "Disease not found"}
    return get_objects_for_subject(disease_uri, sf.treatedBy)

@app.get("/diseases/by-weather")
def get_diseases_by_weather(weather: str):
    weather_uri = find_uri_by_label(weather)
    if not weather_uri:
        return {"error": "Weather not found"}
    return get_values_by_object_property(sf.triggeredBy, weather_uri)

@app.get("/crop/profile")
def get_crop_profile(crop: str):
    crop_uri = find_uri_by_label(crop)
    if not crop_uri:
        return {"error": "Crop not found"}
    return {
        "soil": get_objects_for_subject(crop_uri, sf.grownIn),
        "fertilizer": get_objects_for_subject(crop_uri, sf.receivesFertilizer),
        "diseases": get_objects_for_subject(crop_uri, sf.hasDisease),
        "weather": get_objects_for_subject(crop_uri, sf.hasWeather),
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)
