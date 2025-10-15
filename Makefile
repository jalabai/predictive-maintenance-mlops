# -------- Makefile for Predictive Maintenance Project --------

# Create venv and install dependencies
setup:
	pip install -r requirements.txt

# Run the FastAPI app
run:
	uvicorn app.main:app --reload

# Run tests
test:
	pytest -v

# Train the model
train:
	python src/train.py

# Run Docker container
docker-build:
	docker build -t predictive-maintenance .

docker-run:
	docker run -p 8000:8000 predictive-maintenance

# Clean artifacts and logs
clean:
	rm -rf __pycache__ logs mlruns .pytest_cache
