import joblib
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from .config import RAW_DATA_PATH, TARGET_COL, MODELS_DIR

def main():
    print("[INFO] Loading dataset...")
    df = pd.read_csv(RAW_DATA_PATH)

    # Convert numeric-like columns
    df = df.apply(pd.to_numeric, errors="ignore")

    # Split into X, y
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Keep only numeric columns
    X = X.select_dtypes(include=["number"])
    feature_names = X.columns.tolist()

    print(f"[INFO] Using {len(feature_names)} numeric features for permutation importance analysis")

    # Load model + scaler
    print("[INFO] Loading trained model...")
    model_data = joblib.load(Path(MODELS_DIR) / "baseline_model.pkl")
    model = model_data["model"]
    scaler = model_data["scaler"]

    X_scaled = scaler.transform(X.values)

    print("[INFO] Calculating permutation importance...")
    result = permutation_importance(
        model, X_scaled, y,
        n_repeats=10,
        random_state=42,
        scoring="accuracy"
    )

    importances = result.importances_mean
   
    sorted_idx = importances.argsort()
    top_features = [feature_names[i] for i in sorted_idx]
    top_importances = importances[sorted_idx]

    
    plots_dir = Path(MODELS_DIR).parent / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    plt.figure(figsize=(10, 6))
    plt.barh(top_features[-15:], top_importances[-15:])
    plt.xlabel("Permutation Importance")
    plt.title("Top Features by Permutation Importance")

    output_path = plots_dir / "permutation_importance.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[INFO] Saved permutation importance plot to: {output_path}")

if __name__ == '__main__':
    main()
