from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200

def test_predict_endpoint():
    # First, retrain to ensure model exists
    retrain_data = {
        "temperature": [70, 80, 90],
        "pressure": [100, 110, 120],
        "vibration": [0.5, 0.6, 0.7],
        "failure": [0, 1, 0],
    }
    client.post("/retrain", json=retrain_data)

    # Then, test prediction
    sample_input = {
        "temperature": 80,
        "vibration": 0.5,
        "pressure": 100
    }
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200


def test_retrain_endpoint():
    sample_data = {
        "temperature": [70, 75, 80],
        "pressure": [100, 105, 110],
        "vibration": [0.3, 0.4, 0.5],
        "failure": [0, 1, 0]
    }
    response = client.post("/retrain", json=sample_data)
    assert response.status_code in [200, 202]


