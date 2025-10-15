import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.logger import get_logger

logger = get_logger(__name__)

MODEL_PATH = Path("models/model.pkl")

def train_model(X, y):
    logger.info("Training started...")
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X, y)
    logger.info("Training completed.")
    return model

def load_model():
    if not MODEL_PATH.exists():
        logger.warning(f"Model file not found: {MODEL_PATH}")
        raise FileNotFoundError("Trained model not found.")
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded from {MODEL_PATH}")
    return model



# -----------------------------
# Auto-Retrain Wrapper
# -----------------------------
def retrain_from_processed():
    """Loads processed training data and retrains model."""
    from src.preprocess import load_processed_data  # example
    X, y = load_processed_data()
    return train_model(X, y)
