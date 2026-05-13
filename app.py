# app.py
from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
from pathlib import Path
import pandas as pd
import traceback
from urllib.parse import urlparse
import shap

from src.data_prep import extract_features_from_url

app = Flask(__name__)
app.secret_key = "dev-secret-key"


MODEL_PATH = Path("models") / "sla_model.pkl"
print("[INFO] Loading model from:", MODEL_PATH)

model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
scaler = model_data["scaler"]
model_features = model_data["features"]

# -------------------------------------------------
# SHAP explainer (initialized once)
# -------------------------------------------------
explainer = shap.TreeExplainer(model)

# -------------------------------------------------
# Trusted domains (rule-based override)
# -------------------------------------------------
TRUSTED_DOMAINS_PATH = Path("data") / "trusted_domains.txt"

def load_trusted_domains():
    if not TRUSTED_DOMAINS_PATH.exists():
        return set()
    with open(TRUSTED_DOMAINS_PATH, "r") as f:
        return set(line.strip().lower() for line in f if line.strip())

TRUSTED_DOMAINS = load_trusted_domains()

def is_trusted_domain(url):
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        domain = parsed.netloc.lower()
        return any(domain.endswith(td) for td in TRUSTED_DOMAINS)
    except Exception:
        return False

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url", "").strip()

        if not url:
            flash("Please enter a URL", "warning")
            return redirect(url_for("index"))

        try:
            # -------------------------------------------------
            # RULE-BASED OVERRIDE (trusted domain)
            # -------------------------------------------------
            if is_trusted_domain(url):
                return render_template(
                    "result.html",
                    url=url,
                    prediction="LEGITIMATE",
                    probability=0.99,
                    features=None,
                    reason="Trusted domain (reputation-based override)",
                )

            # -------------------------------------------------
            # ML PIPELINE
            # -------------------------------------------------
            df_feat = extract_features_from_url(url)

            # Ensure feature alignment
            for f in model_features:
                if f not in df_feat.columns:
                    df_feat[f] = 0.0

            df_in = (
                df_feat[model_features]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
            )

            # Scale and predict
            Xs = scaler.transform(df_in.values)
            pred = model.predict(Xs)[0]

            try:
                prob = float(model.predict_proba(Xs)[:, 1][0])
            except Exception:
                prob = None

            # -------------------------------------------------
            # Show ALL extracted features
            # -------------------------------------------------
            all_features = {
                k: float(df_in.iloc[0][k])
                for k in df_in.columns
            }

            return render_template(
                "result.html",
                url=url,
                prediction=str(pred),
                probability=prob,
                features=all_features,
                reason="ML-based detection (SLA-FS++ with explainability)",
            )

        except Exception as e:
            traceback.print_exc()
            flash(f"Error during prediction: {e}", "danger")
            return redirect(url_for("index"))

    return render_template("index.html")

# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
