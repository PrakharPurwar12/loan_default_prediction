# =========================================================
# Loan Default Prediction – FINAL Production Training File
# Models: RandomForest + XGBoost
# =========================================================

# --------------------
# 1. Imports
# --------------------
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --------------------
# 2. Configuration
# --------------------
DATA_PATH = "Loan_default.csv"
RF_MODEL_PATH = "loan_default_rf.pkl"
XGB_MODEL_PATH = "loan_default_xgb.pkl"

TARGET_COL = "Default"
RANDOM_STATE = 42

# --------------------
# 3. Load Dataset
# --------------------
print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)

X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

print("\nTarget Distribution:")
print(y.value_counts())

# --------------------
# 4. Identify Column Types
# --------------------
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

print("\nNumeric Columns:", list(num_cols))
print("Categorical Columns:", list(cat_cols))

# --------------------
# 5. Train–Test Split
# --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# --------------------
# 6. Preprocessing (IMPUTE + SPARSE)
# --------------------
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# --------------------
# 7. Model Definitions
# --------------------
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=14,
    min_samples_split=10,
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE
)

scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

xgb_model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=RANDOM_STATE
)

# --------------------
# 8. Pipelines
# --------------------
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", rf_model)
])

xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", xgb_model)
])

# --------------------
# 9. Train Random Forest
# --------------------
print("\nTraining Random Forest...")
rf_pipeline.fit(X_train, y_train)

rf_preds = rf_pipeline.predict(X_test)
rf_probs = rf_pipeline.predict_proba(X_test)[:, 1]

print("\nRandom Forest Evaluation")
print(confusion_matrix(y_test, rf_preds))
print(classification_report(y_test, rf_preds))

rf_auc = roc_auc_score(y_test, rf_probs)
print("Random Forest ROC-AUC:", rf_auc)

# --------------------
# 10. Train XGBoost
# --------------------
print("\nTraining XGBoost...")
xgb_pipeline.fit(X_train, y_train)

xgb_preds = xgb_pipeline.predict(X_test)
xgb_probs = xgb_pipeline.predict_proba(X_test)[:, 1]

print("\nXGBoost Evaluation")
print(confusion_matrix(y_test, xgb_preds))
print(classification_report(y_test, xgb_preds))

xgb_auc = roc_auc_score(y_test, xgb_probs)
print("XGBoost ROC-AUC:", xgb_auc)

# --------------------
# 11. Model Selection
# --------------------
if xgb_auc > rf_auc:
    best_model = "XGBoost"
else:
    best_model = "Random Forest"

print(f"\nBest Model Selected (ROC-AUC driven): {best_model}")

# --------------------
# 12. Save Models
# --------------------
joblib.dump(rf_pipeline, RF_MODEL_PATH)
joblib.dump(xgb_pipeline, XGB_MODEL_PATH)

print("\nModels saved successfully:")
print(RF_MODEL_PATH)
print(XGB_MODEL_PATH)

# --------------------
# 13. Sanity Check
# --------------------
loaded_model = joblib.load(XGB_MODEL_PATH)
sample_predictions = loaded_model.predict(X_test.head(5))

print("\nSanity Check – Sample Predictions:")
print(sample_predictions)

# =========================================================
# End of File
# =========================================================
