from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import yaml
import mlflow
import sys
from pathlib import Path

# Import logger
from src.logger import get_logger

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.model import train_model, load_model

# Initialize logger
logger = get_logger(__name__)

# -------------------------------
# Initialize FastAPI
# -------------------------------
app = FastAPI(title="Predictive Maintenance API", version="1.0.0")
logger.info("🚀 FastAPI application initialized")

# -------------------------------
# Load Configuration Safely
# -------------------------------
CONFIG_PATH = Path("src/config.yaml")

try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded successfully from {CONFIG_PATH}")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    raise RuntimeError(f"Error loading config.yaml: {e}")

MODEL_PATH = Path(config["model"]["model_path"])
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Load or Initialize Model
# -------------------------------
try:
    model = load_model()
    logger.info("Model loaded successfully.")
except Exception as e:
    model = None
    logger.warning(f"No model found at startup: {e}")

# -------------------------------
# Configure MLflow
# -------------------------------
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])
logger.info("MLflow configured successfully.")

# -------------------------------
# Request Schemas
# -------------------------------
class SensorData(BaseModel):
    temperature: float
    pressure: float
    vibration: float

class RetrainData(BaseModel):
    temperature: list[float]
    pressure: list[float]
    vibration: list[float]
    failure: list[int]

# -------------------------------
# Root Endpoint
# -------------------------------
@app.get("/", response_class=HTMLResponse)
def root():
    logger.info("Root endpoint accessed.")
    return """
    <html>
        <head><title>Predictive Maintenance API</title></head>
        <body style="font-family:Arial; text-align:center; margin-top:80px;">
            <h1>Predictive Maintenance API is Running</h1>
            <p>Visit <a href="/docs">/docs</a> to test the API interactively.</p>
        </body>
    </html>
    """

# -------------------------------
# Prediction Endpoint
# -------------------------------
@app.post("/predict")
def predict(data: SensorData):
    global model

    if model is None:
        logger.error("Prediction failed — model not loaded.")
        raise HTTPException(status_code=500, detail="Model not loaded. Please train or retrain the model first.")

    df = pd.DataFrame([data.model_dump()])
    logger.info(f"Received prediction request: {data.model_dump()}")

    try:
        pred = int(model.predict(df)[0])
        proba = float(model.predict_proba(df)[0][1])
        logger.info(f"Prediction: {pred}, Probability: {proba:.4f}")
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    with mlflow.start_run(run_name="prediction", nested=True):
        mlflow.log_params(data.model_dump())
        mlflow.log_metric("failure_probability", proba)

    return {"failure_prediction": pred, "failure_probability": proba}

# -------------------------------
# Retraining Endpoint
# -------------------------------
@app.post("/retrain")
def retrain(data: RetrainData):
    logger.info("Retraining started.")
    df = pd.DataFrame({
        "temperature": data.temperature,
        "pressure": data.pressure,
        "vibration": data.vibration,
        "failure": data.failure,
    })

    X = df[["temperature", "pressure", "vibration"]]
    y = df["failure"]

    try:
        new_model = train_model(X, y)
        joblib.dump(new_model, MODEL_PATH)
        logger.info(f"New model trained and saved at {MODEL_PATH}")
    except Exception as e:
        logger.exception("Retraining failed")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {e}")

    global model
    model = new_model

    with mlflow.start_run(run_name="retraining", nested=True):
        mlflow.log_param("samples_used", len(df))
        mlflow.sklearn.log_model(new_model, "retrained_model")

    logger.info("Retraining completed successfully.")
    return {"message": "Model retrained successfully", "samples_used": len(df)}

# -------------------------------
# Health Check
# -------------------------------
@app.get("/health")
def health_check():
    status = {"status": "ok", "model_loaded": model is not None}
    logger.info(f"Health check: {status}")
    return status
