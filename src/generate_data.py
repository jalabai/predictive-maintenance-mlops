import pandas as pd
import numpy as np
from pathlib import Path

Path("data/raw").mkdir(parents=True, exist_ok=True)

np.random.seed(42)
n = 1000

data = pd.DataFrame({
    "temperature": np.random.normal(70, 10, n),
    "pressure": np.random.normal(30, 5, n),
    "vibration": np.random.normal(5, 2, n),
    "failure": np.random.choice([0, 1], size=n, p=[0.8, 0.2])
})

data.to_csv("data/raw/sensor_data.csv", index=False)
print(" Synthetic data generated.")
