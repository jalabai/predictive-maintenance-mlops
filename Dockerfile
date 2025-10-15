# Use a slim Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy the project code
COPY . .

# Expose FastAPI (8000) and MLflow (5000)
EXPOSE 8000
EXPOSE 5000

# Default command: run FastAPI + MLflow together
CMD ["bash", "-c", "mlflow ui --host 0.0.0.0 --port 5000 & uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"]
