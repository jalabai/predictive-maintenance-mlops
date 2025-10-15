# Predictive Maintenance MLOps

This repository demonstrates an **end-to-end predictive maintenance workflow** using machine learning and modern MLOps principles. The goal is to predict potential machine failures based on sensor data, deploy predictive models via an API, and automate the full lifecycle of an ML system.

---

##  Features

- **Data Preprocessing**: Clean and transform sensor data for modeling.
- **Model Training**: Train predictive models (Scikit-learn/XGBoost).
- **Inference API**: Serve predictions via FastAPI.
- **Model Retraining**: Update models with new data using API endpoint.
- **Experiment Tracking**: Track runs and model versions with MLflow.
- **CI/CD Automation**: Automated testing, building, and deployment via GitHub Actions.
- **Containerization**: Deployable with Docker.
- **Automated Testing**: Ensure reliability with Pytest.

---

##  Tools & Technologies

- Python 3.11+
- FastAPI (API deployment)
- Pandas (data handling)
- Scikit-learn / XGBoost (model training)
- MLflow (experiment tracking & model registry)
- Docker (containerization)
- GitHub Actions (CI/CD pipeline)
- Pytest (testing)

---

##  Quickstart

1. **Clone the repository**
    ```bash
    git clone https://github.com/jalabai/predictive-maintenance-mlops.git
    cd predictive-maintenance-mlops
    ```

2. **Install dependencies**
    ```bash
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. **Run tests**
    ```bash
    pytest -v -W ignore::DeprecationWarning
    ```

4. **Start the API locally**
    ```bash
    uvicorn app.main:app --reload
    ```
    The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

##  API Endpoints

### `POST /predict`
Predict machine failure from sensor data.

**Request Example**
```json
{
  "temperature": 80,
  "pressure": 100,
  "vibration": 0.5
}
```

**Response Example**
```json
{
  "failure_prediction": 0,
  "failure_probability": 0.02
}
```

---

### `POST /retrain`
Retrain the model with new data.

**Request Example**
```json
{
  "temperature": [80, 85, 78],
  "pressure": [100, 95, 105],
  "vibration": [0.5, 0.7, 0.3],
  "failure": [0, 1, 0]
}
```

**Response Example**
```json
{
  "message": "Model retrained successfully",
  "samples_used": 3
}
```

---

### `GET /health`
Check API and model status.

**Response Example**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

##  CI/CD Pipeline

GitHub Actions automate the following on every push or pull request to the `main` branch:

- Install dependencies
- Run Pytest automated tests
- Build Docker image

On merges to `main`:

- Push Docker image to DockerHub

Ready for deployment to cloud platforms (AWS, Render, etc.).

---

## 🎯Experiment Tracking

- MLflow stores experiment runs and model versions in the `/mlruns` directory.
- For multi-user or cloud deployments, configure MLflow to use a remote tracking server.

---

## Deployment

- Build and run locally with Docker:
    ```bash
    docker build -t predictive-maintenance-api .
    docker run -p 8000:8000 predictive-maintenance-api
    ```
- For cloud deployment, see platform-specific documentation (AWS, Render, etc.).

---

## License

This project is licensed under the [MIT License](LICENSE).

---

##  Notes & Extensibility

- Extend with more complex models, additional sensors, or real-time streaming.
- For production, add authentication and security to API endpoints.
- Contributions and feedback welcome!

---
