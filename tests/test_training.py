from src.model import train_model
import os

def test_train_model_runs(tmp_path):
    from src.model import train_model
    import pandas as pd
    import numpy as np

    X = pd.DataFrame({
        "temperature": np.random.rand(10) * 100,
        "pressure": np.random.rand(10) * 10,
        "vibration": np.random.rand(10)
    })
    y = np.random.randint(0, 2, 10)

    model = train_model(X, y)
    assert model is not None

