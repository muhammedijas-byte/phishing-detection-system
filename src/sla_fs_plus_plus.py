# src/sla_fs_plus_plus.py
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .config import RAW_DATA_PATH, TARGET_COL, MODELS_DIR, RANDOM_STATE
# data_prep imported only for consistency comment; not used directly here
# from .data_prep import prepare_data

# --- user params ---
SAMPLE_ROWS = 500      # SHAP sample size (reduce to speed up)
TOP_K = 20             # how many features to select for SLA-FS++
SHAP_WEIGHT = 0.6      # weight for SHAP in hybrid score (0-1)
# --------------------

def normalize(arr):
    a = np.asarray(arr, dtype=float)
    # avoid division by zero when all values equal
    if np.allclose(a.max(), a.min()):
        return np.zeros_like(a)
    return (a - a.min()) / (a.max() - a.min())

def main():
    print("[INFO] Loading dataset...")
    df = pd.read_csv(RAW_DATA_PATH)
    # Convert numeric-like columns to numbers when possible
    df = df.apply(pd.to_numeric, errors="ignore")

    # Keep numeric columns exactly like training
    X_all = df.drop(columns=[TARGET_COL])
    X_all = X_all.select_dtypes(include=["number"])
    y_all = df[TARGET_COL]

    feature_names = X_all.columns.tolist()
    n_features = len(feature_names)
    print(f"[INFO] Found {n_features} numeric features.")

    print("[INFO] Loading baseline model and scaler...")
    model_data = joblib.load(Path(MODELS_DIR) / "baseline_model.pkl")
    baseline_model = model_data["model"]
    scaler = model_data["scaler"]

    
    X_scaled = scaler.transform(X_all.values)

    print("[INFO] Computing permutation importance (this may take a moment)...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all
    )

    perm_result = permutation_importance(
        baseline_model, X_te, y_te, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1, scoring="accuracy"
    )
    perm_means = np.asarray(perm_result.importances_mean).ravel()  # ensure 1-D

    if perm_means.shape[0] != n_features:
        raise ValueError(f"[ERROR] Permutation importance length {perm_means.shape[0]} != feature count {n_features}")

    perm_dict = dict(zip(feature_names, perm_means))

    print("[INFO] Computing SHAP values (sampled, robust method)...")
    n_rows = X_scaled.shape[0]
    if n_rows > SAMPLE_ROWS:
        rng = np.random.RandomState(RANDOM_STATE)
        sel = rng.choice(n_rows, size=SAMPLE_ROWS, replace=False)
        X_shap = X_scaled[sel]
    else:
        X_shap = X_scaled

    # DataFrame with feature names (important for many SHAP wrappers)
    X_shap_df = pd.DataFrame(X_shap, columns=feature_names)
    print(f"[DEBUG] X_shap_df shape: {X_shap_df.shape}; features expected: {len(feature_names)}")

    shap_means = None

    # --- try modern shap.Explainer first ---
    try:
        expl = shap.Explainer(baseline_model, X_shap_df)
        sv = expl(X_shap_df)  # returns an Explanation object usually
        # many versions: Explanation has .values (array)
        if hasattr(sv, "values"):
            arr = np.asarray(sv.values)
            print(f"[DEBUG] shap.Explainer returned values with shape: {arr.shape}")
            # If arr is (n_samples, n_features)
            if arr.ndim == 2 and arr.shape[1] == len(feature_names):
                shap_means = np.mean(np.abs(arr), axis=0)
            elif arr.ndim == 3:
                # try to collapse last axis: (n_samples, n_features, outputs) -> mean across outputs
                arr2 = arr.mean(axis=2)
                if arr2.shape[1] == len(feature_names):
                    shap_means = np.mean(np.abs(arr2), axis=0)
        # final fallback with sv.values if needed
        if shap_means is None:
            arr_try = np.asarray(sv.values)
            if arr_try.ndim >= 2:
                # try to reshape/collapse to (n_samples, n_features)
                if arr_try.shape[-1] == len(feature_names):
                    # collapse any middle dims by mean
                    arr2 = arr_try.reshape(arr_try.shape[0], -1, arr_try.shape[-1]).mean(axis=1)
                    shap_means = np.mean(np.abs(arr2), axis=0)
    except Exception as e:
        print(f"[WARN] shap.Explainer failed: {e}")

    # --- fallback: TreeExplainer ---
    if shap_means is None:
        try:
            expl2 = shap.TreeExplainer(baseline_model)
            sv2 = expl2.shap_values(X_shap_df)
            print(f"[DEBUG] TreeExplainer returned type: {type(sv2)}")
            if isinstance(sv2, list):
                # choose class 1 if binary classification
                if len(sv2) > 1:
                    arr = np.asarray(sv2[1])
                else:
                    arr = np.asarray(sv2[0])
            else:
                arr = np.asarray(sv2)

            print(f"[DEBUG] TreeExplainer array shape: {arr.shape}")
            if arr.ndim == 2 and arr.shape[1] == len(feature_names):
                shap_means = np.mean(np.abs(arr), axis=0)
            elif arr.ndim == 3:
                if arr.shape[0] == len(feature_names):
                    # unusual order -> transpose then reduce
                    arr2 = arr.transpose(1, 0, 2).mean(axis=2)
                    if arr2.shape[1] == len(feature_names):
                        shap_means = np.mean(np.abs(arr2), axis=0)
                else:
                    arr_mean = np.mean(np.abs(arr), axis=0)
                    if arr_mean.shape[0] == len(feature_names):
                        shap_means = np.mean(np.abs(arr_mean), axis=0)
        except Exception as e:
            print(f"[WARN] TreeExplainer failed: {e}")

    # If still not available, fallback to zeros (script continues)
    if shap_means is None:
        print("[ERROR] Could not compute SHAP per-feature importances reliably. Falling back to zeros for SHAP.")
        shap_means = np.zeros(len(feature_names))

    shap_means = np.asarray(shap_means).ravel()
    if shap_means.shape[0] != len(feature_names):
        raise ValueError(f"[ERROR] Feature length mismatch after SHAP: features={len(feature_names)}, shap={shap_means.shape[0]}")

    shap_dict = dict(zip(feature_names, shap_means))
    print("[INFO] SHAP importances computed.")

    # === 3) Normalize and combine into hybrid score ===
    print("[INFO] Building hybrid ranking (SHAP + Permutation)...")
    perm_norm = normalize([perm_dict[f] for f in feature_names])
    shap_norm = normalize([shap_dict[f] for f in feature_names])

    assert perm_norm.shape == shap_norm.shape == (n_features,)

    hybrid_scores = SHAP_WEIGHT * shap_norm + (1 - SHAP_WEIGHT) * perm_norm

    results_df = pd.DataFrame({
        "feature": feature_names,
        "shap_raw": [shap_dict[f] for f in feature_names],
        "perm_raw": [perm_dict[f] for f in feature_names],
        "shap_norm": shap_norm,
        "perm_norm": perm_norm,
        "hybrid_score": hybrid_scores
    })
    results_df = results_df.sort_values("hybrid_score", ascending=False).reset_index(drop=True)

    # save ranking CSV
    out_dir = Path(MODELS_DIR).parent
    out_dir.mkdir(exist_ok=True, parents=True)
    ranking_csv = out_dir / "sla_fs_plus_plus_ranking.csv"
    results_df.to_csv(ranking_csv, index=False)
    print(f"[INFO] Saved hybrid ranking to: {ranking_csv}")

    # plot top features comparison
    top_n = min(25, len(results_df))
    top_df = results_df.head(top_n).iloc[::-1]  # reversed for horizontal bar

    plt.figure(figsize=(10, max(4, top_n * 0.25)))
    y = np.arange(top_n)
    plt.barh(y - 0.2, top_df["shap_norm"], height=0.4, label="SHAP (norm)")
    plt.barh(y + 0.2, top_df["perm_norm"], height=0.4, label="Perm (norm)")
    plt.yticks(y, top_df["feature"])
    plt.xlabel("Normalized importance")
    plt.title("Top features: SHAP vs Permutation (normalized)")
    plt.legend()
    plt.tight_layout()
    plot_path = out_dir / "plots" / "sla_fs_plus_plus_compare.png"
    plot_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved comparison plot to: {plot_path}")

    # === 4) Select top-K and retrain small model ===
    sel_features = results_df.head(TOP_K)["feature"].tolist()
    print(f"[INFO] Selected top-{TOP_K} features: {sel_features}")

    # prepare X using only selected features (use original df before scaling)
    X_sel = X_all[sel_features]
    # train-test split and scale with new scaler (fit on training)
    X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(
        X_sel, y_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all
    )

    # fit a new scaler (StandardScaler) to selected features
    from sklearn.preprocessing import StandardScaler
    scaler_sel = StandardScaler()
    X_train_sel_scaled = scaler_sel.fit_transform(X_train_sel)
    X_test_sel_scaled = scaler_sel.transform(X_test_sel)

    # train RandomForest on selected features
    print("[INFO] Training RandomForest on selected features...")
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_sel_scaled, y_train_sel)

    train_acc = rf.score(X_train_sel_scaled, y_train_sel)
    test_acc = rf.score(X_test_sel_scaled, y_test_sel)
    print(f"[INFO] Selected-features Train accuracy: {train_acc:.4f}")
    print(f"[INFO] Selected-features Test  accuracy: {test_acc:.4f}")

    # save selected model + scaler + feature list
    out_model_path = Path(MODELS_DIR) / "sla_model.pkl"
    joblib.dump({"model": rf, "scaler": scaler_sel, "features": sel_features}, out_model_path)
    print(f"[INFO] Saved SLA-FS++ model to: {out_model_path}")

    print("[INFO] SLA-FS++ completed.")

if __name__ == "__main__":
    main()
