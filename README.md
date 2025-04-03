An intelligent, n8n-integrated Telegram chatbot that recommends optimal crops and fertilizers based on real-time user inputs like soil nutrients, weather conditions, and pH. This system uses a machine learning backend served through FastAPI and connects seamlessly to a Telegram bot called DSAfarmingbot via an automated n8n workflow.

—

🧠 Overview

This project is designed to support farmers and agricultural decision-makers by providing:

Crop and fertilizer recommendations using trained machine learning models

API-based predictions using FastAPI

n8n workflow automation for user queries and response handling

A fully operational Telegram bot (DSAfarmingbot) for direct user interaction

The entire interaction — from the user typing a query on Telegram to receiving a recommendation — is powered by n8n workflow nodes connected to the FastAPI model backend.

—

🤖 Telegram Bot Integration (DSAfarmingbot)

Users interact via Telegram using @DSAFarmingBot

Example query: "My soil N is 80, P is 45, K is 50, temp 26°C, humidity 85%, pH 6.2, rainfall 120mm"

n8n receives the message via a Telegram Trigger node

Sends the parsed data to the FastAPI endpoint /predict_crop or /predict_fertilizer

Replies back to the user with the predicted result

—

🧪 API Endpoints


GET / → Welcome message
POST /predict_crop → Returns crop name
POST /predict_fertilizer → Returns recommended fertilizer

Sample JSON input:

{ "N": 85, "P": 40, "K": 55, "temperature": 26.5, "humidity": 83.0, "ph": 6.2, "rainfall": 110.0 }

—

⚙️ n8n Workflow Automation

Telegram Trigger → receives user message

Parse & transform message into structured JSON

HTTP Node → calls FastAPI

Send back result to Telegram user via Telegram Node

How to import the workflow:

Go to your n8n instance

Import the workflows/n8n_workflow.json file

Set Telegram credentials and FastAPI URL

Activate workflow

"all the workflows are active, to check the bot you can directly go to the telegram and search for the bot named @DSAFarmingBot"
—

🛠️ Installation & Local Testing

Install dependencies:

pip install -r requirements.txt

Run the FastAPI server:

cd api
uvicorn main:main --reload

Start your n8n service and load the Telegram workflow.

insert the n8n json file and activate the workflow.
—

📈 Evaluation & Error Analysis

Crop classification evaluated on: Accuracy, F1-score, Confusion Matrix

Analysis of misclassified examples


—

📚 Datasets Used

crop_recommendation.csv: Data for crop classification

Fertilizer Prediction.csv: Data for fertilizer matching

—

🌐 Deployment

APIs are deployed on the render.con via github.

if wants to deploy by yourself Use render.yaml and Procfile for automatic builds

Ensure models and dependencies are bundled during deployment
