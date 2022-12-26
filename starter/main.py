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
import pickle
import api_classes

app = FastAPI()


class FakeModel:

    def predict(self, X):
        return X.sum(axis=1) > 2


# load model
with open("model.pkl", "rb") as f:
    model_wrapper = pickle.load(f)


@app.post("/predict")
def predict(item: api_classes.Item):
    """
    request contains JSON with features
    """
    # convert to numpy array
    item_dict = item.dict(by_alias=True)
    # make prediction
    prediction = model_wrapper.predict_single(item_dict)
    # return prediction
    return {"prediction": prediction}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
