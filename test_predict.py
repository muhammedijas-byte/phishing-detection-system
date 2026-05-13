import joblib
import pandas as pd
from pathlib import Path
from src.data_prep import extract_features_from_url

MODEL_PATH = Path("models/sla_model.pkl")

print("[INFO] Loading SLA-FS++ model...")
model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
scaler = model_data["scaler"]
feature_names = model_data["features"]

# Input URL
url = "https://example.com"

# Extract features
df = extract_features_from_url(url)
df = df[feature_names]      # reorder columns to match model
X = scaler.transform(df)

# Predict
pred = model.predict(X)[0]

print("\n=================================")
print(" URL:", url)
print(" Prediction:", pred)
print("=================================")
