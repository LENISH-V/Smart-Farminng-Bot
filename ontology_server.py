from fastapi import FastAPI, Request
from pydantic import BaseModel
from rdflib import Graph, Namespace, RDF, RDFS
import difflib
import uvicorn

app = FastAPI()

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

# Optional for local dev
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

