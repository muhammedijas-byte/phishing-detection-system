import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from .config import RAW_DATA_PATH, TARGET_COL, MODELS_DIR


def main():
    print("[INFO] Loading dataset...")
    df = pd.read_csv(RAW_DATA_PATH)

    # 🔹 NEW: convert numeric-looking columns to numbers
    df = df.apply(pd.to_numeric, errors="ignore")

    # Separate features and label
    X = df.drop(columns=[TARGET_COL])

    # Use ONLY numeric columns (same as prepare_data)
    X = X.select_dtypes(include=["number"])
    feature_names = X.columns.tolist()
    print("[INFO] Using numeric feature columns for SHAP:", feature_names)


    print("[INFO] Loading trained model...")
    model_data = joblib.load(Path(MODELS_DIR) / "baseline_model.pkl")
    model = model_data["model"]
    scaler = model_data["scaler"]

   
    X_scaled = scaler.transform(X.values)

    MAX_ROWS = 500
    if X_scaled.shape[0] > MAX_ROWS:
        rng = np.random.RandomState(42)
        idx = rng.choice(X_scaled.shape[0], size=MAX_ROWS, replace=False)
        X_sample = X_scaled[idx]
    else:
        X_sample = X_scaled

    X_sample_df = pd.DataFrame(X_sample, columns=feature_names)
    print(f"[INFO] Using {len(X_sample_df)} rows for SHAP analysis.")

    print("[INFO] Computing SHAP values (this may take a little time)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample_df)

    # For binary classification, shap_values may be [class0, class1]
    if isinstance(shap_values, list):
        shap_to_plot = shap_values[1]  # class 1 = phishing
    else:
        shap_to_plot = shap_values

    # Create plots folder if not exists
    plots_dir = Path(MODELS_DIR).parent / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # ---------- 1) GLOBAL BAR PLOT ----------
    print("[INFO] Creating SHAP bar plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_to_plot,
        X_sample_df,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    bar_path = plots_dir / "shap_global_bar1.png"
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved bar plot to: {bar_path}")

    # ---------- 2) BEESWARM PLOT ----------
    print("[INFO] Creating SHAP beeswarm plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_to_plot,
        X_sample_df,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    beeswarm_path = plots_dir / "shap_beeswarm1.png"
    plt.savefig(beeswarm_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved beeswarm plot to: {beeswarm_path}")

    print("[INFO] SHAP analysis completed.")


if __name__ == "__main__":
    main()
