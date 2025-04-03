import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset
df = pd.read_csv('Fertilizer Prediction.csv')

# Display first few rows
print(df.head())
print(df.columns.tolist())
df.columns = df.columns.str.strip()


# Features and target selection
# We'll use Soil Type and Crop Type along with environmental factors for prediction.
features = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type']
target = 'Fertilizer'  # This is the fertilizer name.

# Option: if you want to output NPK values too, you can create a mapping from fertilizer name to npk values
npk_mapping = df.groupby('Fertilizer')[['Nitrogen', 'Phosphorous', 'Potassium']].first().to_dict(orient='index')
print("NPK mapping for fertilizers:")
print(npk_mapping)

# Preprocessing pipeline:
# - Numeric features: Temperature, Humidity, Moisture
# - Categorical features: Soil Type, Crop Type
numeric_features = ['Temparature', 'Humidity', 'Moisture']
categorical_features = ['Soil Type', 'Crop Type']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(random_state=42),
    'SVC': SVC(probability=True, random_state=42)
}

# Dictionary to store pipelines and scores
pipelines = {}
scores = {}

for model_name, model in models.items():
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    # Evaluate using cross-validation on training set
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
    avg_score = np.mean(cv_scores)
    scores[model_name] = avg_score
    pipelines[model_name] = pipe
    print(f"{model_name} average CV accuracy: {avg_score:.4f}")

# Select the best model based on CV score
best_model_name = max(scores, key=scores.get)
best_model = pipelines[best_model_name]
print(f"Best model selected: {best_model_name}")

# Train the best model on the entire training set
best_model.fit(X_train, y_train)

# Evaluate on test set
y_pred = best_model.predict(X_test)
print("Test set classification report:")
print(classification_report(y_test, y_pred))

# Error analysis
misclassified_indices = np.where(y_test != y_pred)[0]
misclassified_samples = X_test.iloc[misclassified_indices]
misclassified_labels = y_test.iloc[misclassified_indices]
predicted_labels = pd.Series(y_pred, index=y_test.index).iloc[misclassified_indices]

# Scatter plot of misclassifications
error_analysis_df = pd.DataFrame({
    "Actual Label": misclassified_labels,
    "Predicted Label": predicted_labels
})

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=error_analysis_df,
    x="Actual Label",
    y="Predicted Label",
    hue="Actual Label",
    style="Predicted Label",
    palette="tab10",
    s=100
)
plt.title("Error Analysis - Misclassifications in Fertilizer Recommender")
plt.xlabel("Actual Label")
plt.ylabel("Predicted Label")
plt.legend(title="Misclassification Details", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
os.makedirs("ml_model", exist_ok=True)
plt.savefig("ml_model/misclassification_scatter_plot_updated.png")
plt.show()
print("📊 Updated misclassification scatter plot saved as ml_model/misclassification_scatter_plot_updated.png")

# Train a clustering model on the preprocessed features (using all data)
# For clustering, we use the numerical representation from the preprocessor.
X_processed = preprocessor.fit_transform(X)
# Let's assume 3 clusters for demonstration (this can be tuned)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_processed)
df['Cluster'] = clusters
print("Clustering result (first 10 rows):")
print(df[['Soil Type', 'Crop Type', 'Cluster']].head(10))


# To make a recommendation based on user input, one would:
# 1. Preprocess the input using the preprocessor.
# 2. Predict the fertilizer name.
# 3. Lookup NPK values from the npk_mapping.
def recommend_fertilizer(input_data, npk_mapping):
    """
    input_data should be a dictionary with keys: Temparature, Humidity, Moisture, Soil Type, Crop Type.
    npk_mapping should be a dictionary mapping fertilizer names to NPK values.
    """
    input_df = pd.DataFrame([input_data])
    fert_pred = best_model.predict(input_df)[0]
    npk_values = npk_mapping.get(fert_pred, None)
    return fert_pred, npk_values

# Example usage:
example_input = {
    'Temparature': 30,
    'Humidity': 55,
    'Moisture': 40,
    'Soil Type': 'Loamy',
    'Crop Type': 'Sugarcane'}
fertilizer, npk = recommend_fertilizer(example_input, npk_mapping)
fertilizer, npk = recommend_fertilizer(example_input, npk_mapping)
print("Recommended Fertilizer:", fertilizer)
print("NPK Values:", npk)
