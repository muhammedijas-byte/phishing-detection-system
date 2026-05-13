# check_model_features.py
import joblib
m = joblib.load("models/sla_model.pkl")
print("Model 'features' list:", m.get("features"))
