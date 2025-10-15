# src/preprocess.py
import pandas as pd
import yaml
from pathlib import Path

# Load config
with open("src/config.yaml") as f:
    config = yaml.safe_load(f)

raw_path = Path(config["data"]["raw_path"])
processed_path = Path(config["data"]["processed_path"])
processed_path.parent.mkdir(parents=True, exist_ok=True)

# Load raw data
df = pd.read_csv(raw_path)

# Example preprocessing
df = df.dropna()  # remove missing values
# Normalize numeric features
for col in ["temperature", "pressure", "vibration"]:
    df[col] = (df[col] - df[col].mean()) / df[col].std()

# Save processed data
df.to_csv(processed_path, index=False)
print(f"Preprocessed data saved to {processed_path}")
