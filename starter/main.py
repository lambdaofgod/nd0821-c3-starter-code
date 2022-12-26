"""
main.py

FastAPI api that serves a machine learning model
model is loaded from "model.pkl"

Endpoints:
- predict (POST)
    request contains JSON with features
"""

import uvicorn
from fastapi import FastAPI
import pandas as pd


app = FastAPI()


class FakeModel:

    def predict(self, X):
        return X.sum(axis=1) > 2


# load model
# with open("model.pkl", "rb") as f:
#    model = pickle.load(f)

model = FakeModel()


@app.post("/predict")
def predict(features: dict):
    """
    request contains JSON with features
    """
    # convert to numpy array
    features_df = pd.DataFrame.from_records([features])
    # make prediction
    prediction = model.predict(features_df)[0]
    # return prediction
    return {"prediction": int(prediction)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
