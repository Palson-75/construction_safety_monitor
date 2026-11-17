
from pathlib import Path
import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT / "outputs" / "full_video_ppe"
    / "final_risk_dataset.csv"
)

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "xgboost_model.pkl"
FEATURE_FILE = MODEL_DIR / "xgboost_features.pkl"

# Load dataset
df = pd.read_csv(DATA_FILE)

# Keep only records with definite labels
df = df[df["risk_level"].isin(["Low", "High"])].copy()

print("\n========== TRAINING DATA ==========")
print("Total labelled records:", len(df))
print(df["risk_level"].value_counts())

if df["risk_level"].nunique() < 2:
    raise ValueError("Both Low and High classes are required.")

if df["risk_level"].value_counts().min() < 2:
    raise ValueError("Not enough samples in each class.")

# Numerical features only.
# Exclude IDs, video names, status columns and target.
excluded = {
    "video_name",
    "worker_id",
    "risk_level"
}

feature_columns = [
    col for col in df.columns
    if col not in excluded
    and not col.endswith("_status")
    and pd.api.types.is_numeric_dtype(df[col])
]

X = df[feature_columns].fillna(0)
y = df["risk_level"].map({"Low": 0, "High": 1})

# Split while preserving class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.50,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# Small model to reduce overfitting
model = XGBClassifier(
    n_estimators=40,
    max_depth=2,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

print("\nTraining XGBoost...")
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

print("\n========== MODEL EVALUATION ==========")
print("Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Low", "High"],
        zero_division=0
    )
)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# Save model and exact feature order
joblib.dump(model, MODEL_FILE)
joblib.dump(feature_columns, FEATURE_FILE)

print("\n========== MODEL SAVED ==========")
print("Model:", MODEL_FILE)
print("Features:", FEATURE_FILE)
print("Feature count:", len(feature_columns))
