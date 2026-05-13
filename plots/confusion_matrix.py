import sys
from pathlib import Path

# -------------------------------------------------
# Fix import path so 'src' works
# -------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.data_prep import load_data, prepare_data

# -------------------------------------------------
# 1. Load dataset
# -------------------------------------------------
df = load_data()
X_train, X_test, y_train, y_test, scaler = prepare_data(df)

# -------------------------------------------------
# 2. Load final SLA model
# -------------------------------------------------
model_data = joblib.load(ROOT / "models" / "sla_model.pkl")
model = model_data["model"]
sel_features = model_data["features"]

# -------------------------------------------------
# 3. Convert test set to DataFrame with names
# -------------------------------------------------
X_test_df = pd.DataFrame(X_test, columns=scaler.feature_names_in_)

# -------------------------------------------------
# 4. Scale ALL features (same as training)
# -------------------------------------------------
X_test_scaled_all = scaler.transform(X_test_df)

X_test_scaled_df = pd.DataFrame(
    X_test_scaled_all,
    columns=scaler.feature_names_in_
)

# -------------------------------------------------
# 5. Select SLA features
# -------------------------------------------------
X_test_scaled_sel = X_test_scaled_df[sel_features]

# -------------------------------------------------
# 6. Predict
# -------------------------------------------------
preds = model.predict(X_test_scaled_sel.values)

# -------------------------------------------------
# 7. Confusion Matrix
# -------------------------------------------------
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix – SLA-FS++ Model")

plots_dir = ROOT / "plots"
plots_dir.mkdir(exist_ok=True)

plt.savefig(plots_dir / "confusion_matrix.png", dpi=300)
plt.close()

print("[INFO] Confusion matrix saved to plots/confusion_matrix.png")
