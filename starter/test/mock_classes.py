import numpy as np


class MockModel:

    def predict(self, X):
        return X.max(axis=1) > 2

    def predict_proba(self, X):
        y = np.zeros((X.shape[0], 2))
        y[:, 1] = 1
        return y


class MockEncoder:

    def transform(self, X):
        return X
