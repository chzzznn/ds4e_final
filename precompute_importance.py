from __future__ import annotations
 
import pickle
import time
from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
 

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
 
DATASET_PATH = "Global_Education.csv"
 
FEATURES = [
    "Completion_Rate_Primary_Male",
    "Completion_Rate_Primary_Female",
    "Completion_Rate_Lower_Secondary_Male",
    "Completion_Rate_Lower_Secondary_Female",
    "Completion_Rate_Upper_Secondary_Male",
    "Completion_Rate_Upper_Secondary_Female",
    "Youth_15_24_Literacy_Rate_Male",
    "Youth_15_24_Literacy_Rate_Female",
    "Birth_Rate",
    "Gross_Primary_Education_Enrollment",
    "Gross_Tertiary_Education_Enrollment",
    "Lower_Secondary_End_Proficiency_Reading",
    "Lower_Secondary_End_Proficiency_Math",
]
 
TARGET = "Unemployment_Rate"
 
MODELS = {
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": lambda: GradientBoostingRegressor(
        n_estimators=100, random_state=42
    ),
}
 

def load_data() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = pd.read_csv(DATASET_PATH, encoding="latin1")
    df.columns = df.columns.str.strip()
 
    df[TARGET] = df[TARGET].replace(0, np.nan)
    df_model = df[FEATURES + [TARGET]].copy()
    df_model = df_model.replace(0, np.nan)
    df_model = df_model.dropna(subset=[TARGET])
 
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(
        imputer.fit_transform(df_model[FEATURES]), columns=FEATURES
    )
    y = df_model[TARGET].values
 
    return X, y, FEATURES
 
 
def cache_path(model_name: str) -> Path:
    safe_model = model_name.lower().replace(" ", "_")
    return CACHE_DIR / f"importance_{safe_model}.pkl"
 
 
def compute_for(model_name: str) -> dict:
    X, y, features = load_data()
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
 
    model = MODELS[model_name]()
    model.fit(X_train, y_train)
 

    imp_df = pd.DataFrame(
        {
            "Feature": features,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=True)
 

    perm_result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    perm_df = pd.DataFrame(
        {
            "Feature": features,
            "Importance": perm_result.importances_mean,
            "Std": perm_result.importances_std,
        }
    ).sort_values("Importance", ascending=True)
 
    payload = {
        "features": features,
        "imp_df": imp_df,
        "perm_df": perm_df,
        "X_test": X_test.reset_index(drop=True),
    }
 

    try:
        import shap
 
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        payload["shap_values"] = np.asarray(shap_values)
        payload["shap_df"] = pd.DataFrame(
            {
                "Feature": features,
                "Mean |SHAP|": np.abs(shap_values).mean(axis=0),
            }
        ).sort_values("Mean |SHAP|", ascending=True)
    except ImportError:
        print("  (shap not installed — skipping SHAP)")
 
    return payload
 
 
def main() -> None:
    print(f"Cache directory: {CACHE_DIR}")
    for model_name in MODELS:
        t0 = time.time()
        print(f"→ {model_name}")
        payload = compute_for(model_name)
        out = cache_path(model_name)
        with out.open("wb") as fh:
            pickle.dump(payload, fh)
        print(f"  saved {out.name} ({time.time() - t0:.1f}s)")
    print("Done.")
 
 
if __name__ == "__main__":
    main()