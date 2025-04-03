import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ========== Load Dataset ==========
try:
    df = pd.read_csv("crop_recommendation.csv")
    print("Dataset Loaded. Shape:", df.shape)
    print("Features:", list(df.columns))
except FileNotFoundError:
    print("Error: Dataset file 'crop_recommendation.csv' not found.")
    exit()

# ========== Step 2: Features and Target ==========
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
if not all(feature in df.columns for feature in features + ['label']):
    print("Error: Missing required columns in the dataset.")
    exit()

X = df[features]
y = df['label']

# ========== Step 3: Train/Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== Step 4: Initialize Models ==========
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=500)
}

best_model = None
best_score = 0
results = ""

for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results += f"\n{name} Accuracy: {acc:.4f}\n"
        results += classification_report(y_test, y_pred)
        
        print(f" {name} trained. Accuracy: {acc:.4f}")
        
        if acc > best_score:
            best_score = acc
            best_model = model
    except Exception as e:
        print(f"Error training {name}: {e}")

# ========== Step 7: Confusion Matrix ==========
if best_model:
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Confusion Matrix of Best Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    os.makedirs("ml_model", exist_ok=True)
    plt.savefig("ml_model/confusion_matrix.png")
    print("📊 Confusion matrix saved as ml_model/confusion_matrix.png")
else:
    print("No model was successfully trained.")

# ========== Step 8: Save Evaluation Report ==========
try:
    with open("ml_model/evaluation_report.txt", "w") as report_file:
        report_file.write(results)
    print("📄 Evaluation report saved as ml_model/evaluation_report.txt")
except Exception as e:
    print(f"Error saving evaluation report: {e}")

# ========== Step 9: Error Analysis ==========
if best_model:
    misclassified_indices = np.where(y_test != y_pred_best)[0]
    misclassified_samples = X_test.iloc[misclassified_indices]
    misclassified_labels = y_test.iloc[misclassified_indices]
    predicted_labels = pd.Series(y_pred_best, index=y_test.index).iloc[misclassified_indices]

    # Visualize misclassifications
    error_analysis_df = pd.DataFrame({
        "Actual Label": misclassified_labels,
        "Predicted Label": predicted_labels
    })

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=misclassified_samples['temperature'], 
        y=misclassified_samples['humidity'], 
        hue=error_analysis_df['Actual Label'] + " -> " + error_analysis_df['Predicted Label'],
        palette="tab10"
    )
    plt.title("Misclassifications Visualization")
    plt.xlabel("Temperature")
    plt.ylabel("Humidity")
    plt.legend(title="Actual -> Predicted", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    os.makedirs("ml_model", exist_ok=True)
    plt.savefig("ml_model/misclassifications.png")
    plt.show()
    print("📊 Misclassifications visualization saved as ml_model/misclassifications.png")
else:
    print("❌ Error analysis skipped as no model was successfully trained.")



