import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from .data_prep import load_data, prepare_data
from .config import MODELS_DIR


def main():
    # 1. Load dataset
    df = load_data()

    # 2. Prepare data (split + scale)
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    # 3. Train Random Forest
    print("[INFO] Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # 4. Test model
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"[INFO] Accuracy: {acc:.4f}")

    # 5. Save the model & scaler
    save_path = Path(MODELS_DIR) / "baseline_model.pkl"
    joblib.dump({"model": model, "scaler": scaler}, save_path)
    print("[INFO] Model saved to:", save_path)


if __name__ == "__main__":
    main()
