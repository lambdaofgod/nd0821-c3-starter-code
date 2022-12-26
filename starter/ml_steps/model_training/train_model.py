# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml_steps.data import process_data
from ml_steps.model_training import model
import pandas as pd
import pickle

# Add the necessary imports for the starter code.


# Add code to load in the data.
data = pd.read_csv("data/census_clean.csv")

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, feature_encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
(X_test, y_test, __,  __) = process_data(
    test,
    label="salary",
    feature_encoder=feature_encoder,
    lb=lb
)

# Train and save a model.

trained_model = model.train_model(X_train, y_train)

with open("model.pkl", "wb") as f:
    pickle.dump(trained_model, f)
