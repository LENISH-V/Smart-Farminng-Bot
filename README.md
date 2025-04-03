# Smart-Farminng-Bot

An intelligent and user-friendly chatbot system that recommends optimal crops and fertilizers based on environmental conditions using machine learning. The system is API-driven, scalable, and designed to be integrated with platforms like N8N or Dialogflow for chatbot automation.
DSA_Group_9

Lakshika Paiva
Sheryl Shajan
Lenish Vaghasiya

Overview

This project combines machine learning and conversational automation to assist farmers in choosing the most suitable crop and fertilizer. It features:

A FastAPI backend for crop and fertilizer recommendation

Pre-trained classification models using soil and weather parameters

REST API endpoints for programmatic access

Deployment-ready setup for platforms like Render

Supports integration with tools like N8N and Dialogflow for building intelligent chatbot workflows

—

Project Structure

. ├── api/ → FastAPI backend
│ ├── main.py → Core API routes
│ └── models.py → Pydantic schema
├── ml_model/ → Trained models (crop & fertilizer)
│ └── best_fert.pkl
├── Data/ → CSV datasets
│ ├── crop_recommendation.csv
│ └── Fertilizer Prediction.csv
├── requirements.txt → Project dependencies
├── render.yaml → Render deployment config
└── procfile → Deployment process config

—

API Endpoints

Base URL: http://localhost:8000

GET / → Welcome message
POST /predict_crop → Returns recommended crop based on input
POST /predict_fertilizer → Returns recommended fertilizer

Sample JSON Payload for /predict_crop:

{ "N": 90, "P": 40, "K": 45, "temperature": 27.0, "humidity": 80.0, "ph": 6.5, "rainfall": 200.0 }

—

Installation

Clone the repo and install dependencies:

pip install -r requirements.txt

Run the FastAPI app:

cd api
uvicorn main:main --reload

—

Deployment

This project is ready to deploy on Render:

Configure service using render.yaml

Use Procfile for production command

Ensure your model files and data are bundled

—

Evaluation & Error Analysis

ML model performance metrics: Accuracy, Precision, Recall, F1-score

Confusion matrix for crop prediction

Analysis of misclassified inputs

Visualizations and logs for model explainability

—

Datasets

crop_recommendation.csv: Contains data on nutrients, temperature, humidity, pH, rainfall, and crop label

Fertilizer Prediction.csv: Dataset mapping crop condition to optimal fertilizer
