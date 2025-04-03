
from fastapi import FastAPI
from pydantic import BaseModel
from rdflib import Graph, Namespace, RDF, RDFS
import difflib
import uvicorn

app = FastAPI()

# Load TTL ontology
g = Graph()
g.parse("https://smartfarminng-bot.onrender.com/static/smart_farming_enriched.ttl", format="ttl")

sf = Namespace("http://www.smartfarming.org/ontology#")

class Query(BaseModel):
    text: str

def get_labels_by_class(klass):
    return [
        str(s.split("#")[-1])
        for s in g.subjects(RDF.type, sf[klass])
    ]

def fuzzy_match(text, choices, cutoff=0.6):
    match = difflib.get_close_matches(text, choices, n=1, cutoff=cutoff)
    return match[0] if match else None

@app.post("/match")
async def match_query(query: Query):
    text = query.text.lower()

    crops = get_labels_by_class("Crop")
    ferts = get_labels_by_class("Fertilizer")
    soils = get_labels_by_class("Soil")
    weathers = get_labels_by_class("Weather")
    diseases = get_labels_by_class("Disease")

    matched_crop = fuzzy_match(text, crops)
    matched_soil = fuzzy_match(text, soils)
    matched_fert = fuzzy_match(text, ferts)
    matched_weather = fuzzy_match(text, weathers)
    matched_disease = fuzzy_match(text, diseases)

    response = {
        "input": text,
        "matched_crop": matched_crop,
        "matched_soil": matched_soil,
        "matched_fertilizer": matched_fert,
        "matched_weather": matched_weather,
        "matched_disease": matched_disease,
        "recommendations": []
    }

    if matched_crop:
        crop_uri = sf[matched_crop]
        fert = g.value(crop_uri, sf["receivesFertilizer"])
        soil = g.value(crop_uri, sf["grownIn"])
        weather = g.value(crop_uri, sf["hasWeather"])
        disease = g.value(crop_uri, sf["hasDisease"])

        if fert:
            response["recommendations"].append(f"Crop '{matched_crop}' uses fertilizer: {fert.split('#')[-1]}")
        if soil:
            response["recommendations"].append(f"Crop '{matched_crop}' grows in: {soil.split('#')[-1]}")
        if weather:
            response["recommendations"].append(f"Crop '{matched_crop}' prefers weather: {weather.split('#')[-1]}")
        if disease:
            response["recommendations"].append(f"Crop '{matched_crop}' may be affected by: {disease.split('#')[-1]}")
            treat = g.value(disease, sf["treatedBy"])
            symptom = g.value(disease, sf["showsSymptom"])
            trigger = g.value(disease, sf["triggeredBy"])
            if treat:
                response["recommendations"].append(f"Disease '{disease.split('#')[-1]}' is treated by: {treat.split('#')[-1]}")
            if symptom:
                response["recommendations"].append(f"Disease '{disease.split('#')[-1]}' shows symptom: {symptom.split('#')[-1]}")
            if trigger:
                response["recommendations"].append(f"Disease '{disease.split('#')[-1]}' is triggered by: {trigger.split('#')[-1]}")

    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
