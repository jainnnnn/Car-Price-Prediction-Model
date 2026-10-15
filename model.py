import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

PIPELINE_PATH = "model_pipeline.pkl"
META_PATH = "model_meta.json"

df = pd.read_csv("CarPrice_Assignment.csv")
print(df.head())

# Misspelled brand names found in the CarName column
BRAND_FIXES = {
    "maxda": "mazda",
    "porcshce": "porsche",
    "toyouta": "toyota",
    "vokswagen": "volkswagen",
    "vw": "volkswagen",
}

# car_ID is just a row number, so drop it
df = df.drop(columns=["car_ID"])

# CarName = "<brand> <model>". Keep the brand, drop the rest.
df["brand"] = (
    df["CarName"].str.strip().str.split().str[0].str.lower().replace(BRAND_FIXES)
)
df = df.drop(columns=["CarName"])

# Price is right-skewed, so predict log(1 + price)
df["price_log"] = np.log1p(df["price"])

# Split X and y
X = df.drop(columns=["price", "price_log"])
y = df["price_log"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# Preprocessing: scale numbers, one-hot encode categories
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# Create the pipeline
gb_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ]
)

# Train the model
gb_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = gb_pipeline.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mae_log = mean_absolute_error(y_test, y_pred)
rmse_log = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae_usd = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
cv_r2 = cross_val_score(gb_pipeline, X, y, cv=5, scoring="r2")

print("Gradient Boosting Model Performance (test split)")
print(f"R^2 Score     : {r2:.4f}")
print(f"MAE (log)     : {mae_log:.4f}")
print(f"RMSE (log)    : {rmse_log:.4f}")
print(f"MAE (USD)     : {mae_usd:,.0f}")
print(f"5-fold CV R^2 : {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")


# Save the fitted pipeline object
with open(PIPELINE_PATH, "wb") as f:
    pickle.dump(gb_pipeline, f)

meta = {
    "n_rows": int(len(df)),
    "metrics": {
        "r2": float(r2),
        "mae_usd": float(mae_usd),
        "rmse_log": rmse_log,
        "cv_r2_mean": float(cv_r2.mean()),
    },
    "price": {
        "min": float(df["price"].min()),
        "median": float(df["price"].median()),
        "max": float(df["price"].max()),
    },
    "numeric": {
        col: {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "median": float(X[col].median()),
            "is_int": bool(pd.api.types.is_integer_dtype(X[col])),
        }
        for col in numerical_cols
    },
    "categorical": {
        col: {
            "options": sorted(X[col].astype(str).unique().tolist()),
            "default": str(X[col].mode().iloc[0]),
        }
        for col in categorical_cols
    },
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print(f"Saved {PIPELINE_PATH} and {META_PATH}")