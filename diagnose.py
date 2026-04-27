import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer

# ── Load exactly as the app does ──────────────────────────
df = pd.read_csv("Global_Education.csv", encoding="latin1")
df.columns = df.columns.str.strip()
df["Unemployment_Rate"] = df["Unemployment_Rate"].replace(0, np.nan)

features = [
    "Completion_Rate_Primary_Male", "Completion_Rate_Primary_Female",
    "Completion_Rate_Lower_Secondary_Male", "Completion_Rate_Lower_Secondary_Female",
    "Completion_Rate_Upper_Secondary_Male", "Completion_Rate_Upper_Secondary_Female",
    "Youth_15_24_Literacy_Rate_Male", "Youth_15_24_Literacy_Rate_Female",
    "Birth_Rate", "Gross_Primary_Education_Enrollment",
    "Gross_Tertiary_Education_Enrollment",
    "Lower_Secondary_End_Proficiency_Reading", "Lower_Secondary_End_Proficiency_Math",
]

df_model = df[features + ["Unemployment_Rate", "Countries and areas"]].copy()
df_model = df_model.replace(0, np.nan)
df_model = df_model.dropna(subset=["Unemployment_Rate"])

print("=" * 55)
print(f"  Rows after filtering:     {len(df_model)}")
print(f"  Unemployment range:       {df_model['Unemployment_Rate'].min():.1f}% – {df_model['Unemployment_Rate'].max():.1f}%")
print(f"  Unemployment mean:        {df_model['Unemployment_Rate'].mean():.2f}%")
print(f"  Unemployment std:         {df_model['Unemployment_Rate'].std():.2f}%")
print("=" * 55)

# ── Check NaN counts per feature ─────────────────────────
print("\nNaN counts per feature (before imputation):")
for f in features:
    n = df_model[f].isna().sum()
    pct = n / len(df_model) * 100
    flag = " ⚠️  HIGH" if pct > 40 else ""
    print(f"  {f:<52} {n:>3} ({pct:.0f}%){flag}")

# ── WRONG way (current app) ───────────────────────────────
print("\n" + "=" * 55)
print("  CURRENT APP PIPELINE (impute first, split after)")
print("=" * 55)
imputer_bad = SimpleImputer(strategy="median")
X_bad = pd.DataFrame(imputer_bad.fit_transform(df_model[features]), columns=features)
y = df_model["Unemployment_Rate"].values

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_bad, y, test_size=0.2, random_state=42)
scaler_bad = StandardScaler()
X_train_bs = scaler_bad.fit_transform(X_train_b)
X_test_bs  = scaler_bad.transform(X_test_b)

for name, model in [
    ("Linear Regression",  LinearRegression()),
    ("Random Forest",      RandomForestRegressor(n_estimators=100, random_state=42)),
    ("Gradient Boosting",  GradientBoostingRegressor(n_estimators=100, random_state=42)),
]:
    model.fit(X_train_bs, y_train_b)
    preds = model.predict(X_test_bs)
    print(f"  {name:<22}  R²={r2_score(y_test_b, preds):+.3f}  RMSE={np.sqrt(mean_squared_error(y_test_b, preds)):.2f}")

# ── CORRECT way (split first, impute after) ───────────────
print("\n" + "=" * 55)
print("  FIXED PIPELINE (split first, impute after)")
print("=" * 55)
X_raw = df_model[features].copy()
y     = df_model["Unemployment_Rate"].values

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_raw, y, test_size=0.2, random_state=42)

imputer_good = SimpleImputer(strategy="median")
X_train_r = pd.DataFrame(imputer_good.fit_transform(X_train_r), columns=features)
X_test_r  = pd.DataFrame(imputer_good.transform(X_test_r),      columns=features)

scaler_good = StandardScaler()
X_train_rs = scaler_good.fit_transform(X_train_r)
X_test_rs  = scaler_good.transform(X_test_r)

for name, model in [
    ("Linear Regression",  LinearRegression()),
    ("Random Forest",      RandomForestRegressor(n_estimators=100, random_state=42)),
    ("Gradient Boosting",  GradientBoostingRegressor(n_estimators=100, random_state=42)),
]:
    model.fit(X_train_rs, y_train_r)
    preds = model.predict(X_test_rs)
    print(f"  {name:<22}  R²={r2_score(y_test_r, preds):+.3f}  RMSE={np.sqrt(mean_squared_error(y_test_r, preds)):.2f}")

# ── y distribution check ──────────────────────────────────
print("\n" + "=" * 55)
print("  y_test distribution check")
print("=" * 55)
_, _, _, y_test_check = train_test_split(X_raw, y, test_size=0.2, random_state=42)
print(f"  y_test  mean={y_test_check.mean():.2f}  std={y_test_check.std():.2f}  "
      f"min={y_test_check.min():.1f}  max={y_test_check.max():.1f}  n={len(y_test_check)}")
print(f"  Baseline (predict mean always): "
      f"R²={r2_score(y_test_check, np.full_like(y_test_check, y_test_check.mean())):.3f}")
