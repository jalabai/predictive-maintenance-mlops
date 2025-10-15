# src/train.py
import pandas as pd
import yaml
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import joblib
import mlflow
import mlflow.sklearn
from src.model import save_model

# Load config
with open("src/config.yaml") as f:
    config = yaml.safe_load(f)

processed_path = Path(config["data"]["processed_path"])
model_path = Path(config["model"]["model_path"])

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])

# --- Load processed data ---
df = pd.read_csv(processed_path)

# Features and target
X = df[["temperature", "pressure", "vibration"]]
y = df["failure"]  # make sure your CSV has a 'failure' column

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Train model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Training completed. F1 Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

# --- Save model ---
save_model(model, model_path)

# --- Log to MLflow ---
with mlflow.start_run(run_name="train_model"):
    mlflow.log_params({"n_estimators": 100, "random_state": 42})
    mlflow.log_metrics({"f1_score": f1, "roc_auc": roc_auc})
    mlflow.sklearn.log_model(model, artifact_path="model")

print(f"Model logged to MLflow and saved at {model_path}")
