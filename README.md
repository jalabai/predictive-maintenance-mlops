Predictive Maintenance MLOps

This project demonstrates an "end-to-end predictive maintenance workflow" using machine learning and MLOps principles. The goal is to predict potential machine failures based on sensor data, deploy the trained model with FastAPI, track experiments with MLflow, and automate testing, building, and deployment using GitHub Actions.

The project is designed to showcase the full lifecycle of a production-ready ML system: "data preprocessing, model training, inference, retraining, experiment tracking, and CI/CD automation"

Tools & Technologies
- Python 3.11
- FastAPI – for API deployment
- Pandas – data handling
- Scikit-learn / XGBoost – model training
- MLflow – experiment tracking and model versioning
- Docker – containerization
- GitHub Actions– CI/CD automation
- Pytest– automated testing

Setup Guide

1. Clone the repository
git clone https://github.com/jalabai/predictive-maintenance-mlops.git
cd predictive-maintenance-mlops
2. Install dependencies
bash
Copy code
python -m pip install --upgrade pip
pip install -r requirements.txt
3. Run tests
bash
Copy code
pytest -v -W ignore::DeprecationWarning
4. Run the FastAPI app locally
bash
Copy code
uvicorn app.main:app --reload
The API will be available at http://127.0.0.1:8000.

API Endpoints
POST /predict
Predict machine failure from sensor data.

Request JSON Example:
{
  "temperature": 80,
  "pressure": 100,
  "vibration": 0.5
}
Response Example:

{
  "failure_prediction": 0,
  "failure_probability": 0.02
}
POST /retrain
Retrain the model with new data.

Request JSON Example:

{
  "temperature": [80, 85, 78],
  "pressure": [100, 95, 105],
  "vibration": [0.5, 0.7, 0.3],
  "failure": [0, 1, 0]
}
Response Example:

{
  "message": "Model retrained successfully",
  "samples_used": 3
}
GET /health
Check API and model status.

Response Example:


{
  "status": "ok",
  "model_loaded": true
}

CI/CD Pipeline
The repository is configured with GitHub Actions to automate testing, building, and deployment:

On every push or pull request to the main branch:

Install dependencies

Run automated tests with Pytest

Build a Docker image

On main branch only:

Push Docker image to DockerHub

Ready for deployment to cloud platforms (AWS, Render, etc.)

License
This project is licensed under the MIT License. See the LICENSE file for details.

Notes
Experiment tracking and model versions are stored in the /mlruns directory using MLflow.

The project can be extended with more complex models or deployed to cloud infrastructure for real-time monitoring.
