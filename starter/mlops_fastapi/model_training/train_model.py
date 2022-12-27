# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from mlops_fastapi.data import process_data
from mlops_fastapi.model_training import model
from mlops_fastapi import config
import pandas as pd
import pickle


# Add the necessary imports for the starter code.

def main(training_config: config.TrainingConfig,
         data_path="data/census_clean.csv"):
    data = pd.read_csv(data_path)
    train, test = train_test_split(data, test_size=training_config.test_size,
                                   random_state=training_config.test_random_state)

    X_train, y_train, feature_encoder, lb = process_data(
        train, categorical_features=training_config.cat_features, label="salary", training=True)

    # Proces the test data with the process_data function.
    (X_test, y_test, __, __) = process_data(
        test,
        label="salary",
        feature_encoder=feature_encoder,
        lb=lb
    )

    # Train and save a model.
    trained_model = model.train_model(X_train, y_train)
    model_wrapper = model.ClassifierWrapper(trained_model, feature_encoder, lb)
    with open("model.pkl", "wb") as f:
        pickle.dump(model_wrapper, f)


if __name__ == "__main__":
    train_config = config.TrainingConfig
    data_path = "data/census_clean.csv"
    main(train_config, data_path)
