"""
main.py

FastAPI api that serves a machine learning model
model is loaded from "model.pkl"

Endpoints:
- predict (POST)
    request contains JSON with features
"""

from enum import Enum
import uvicorn
from fastapi import FastAPI
import pickle
import api_classes

app = FastAPI()


class PredictionType(str, Enum):
    result = "result"
    probability = "probability"


# load model
with open("model.pkl", "rb") as f:
    model_wrapper = pickle.load(f)


@app.get("/root")
def root():
    return {"message": "Hello World"}


@app.post("/predict/{prediction_type}")
def predict(item: api_classes.Item, prediction_type: PredictionType):
    """
    request contains JSON with features
    """
    assert prediction_type in {"result", "probability"}
    # convert to numpy array
    item_dict = item.dict(by_alias=True)
    # make prediction
    if prediction_type is PredictionType.result:
        prediction = model_wrapper.predict_single(item_dict)
        # return prediction
        return {"prediction": prediction}
    prediction = model_wrapper.predict_single_proba(item_dict)
    return prediction


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
