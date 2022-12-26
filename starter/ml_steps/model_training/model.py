from dataclasses import dataclass

import pandas as pd
from sklearn import base, ensemble, pipeline
from sklearn.metrics import fbeta_score, precision_score, recall_score


@dataclass
class ClassifierWrapper:
    """
    class wrapping classifier pipeline and target encoder
    """

    clf: base.ClassifierMixin
    feature_encoder: base.TransformerMixin
    target_encoder: base.TransformerMixin

    def predict_single(self, item_dict):
        single_item_df = pd.DataFrame.from_records([item_dict])
        features = self.feature_encoder.transform(single_item_df)
        return self.target_encoder.inverse_transform(
            self.clf.predict(features))[0]

# Optional: implement hyperparameter tuning.


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = ensemble.RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds, average="micro"):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1, average=average)
    precision = precision_score(y, preds, zero_division=1, average=average)
    recall = recall_score(y, preds, zero_division=1, average=average)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)
