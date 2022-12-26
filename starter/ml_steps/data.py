import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn import compose


def process_data(
        X,
        categorical_features=[],
        label=None,
        training=True,
        feature_encoder=None,
        lb=None):
    """Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : column transformer that encodes categorical_features with OneHotEncoding and does not touch other features
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])
    num_features = [col for col in X.columns if not col ==
                    label and col not in categorical_features]
    feature_encoder = compose.ColumnTransformer(
        [("numerical", "passthrough", num_features),
         ("categorical", OneHotEncoder(sparse=False), categorical_features)]
    )
    if training is True:
        lb = LabelBinarizer()
        y = lb.fit_transform(y.values).ravel()
        X_features = feature_encoder.fit_transform(X)
    else:
        X_features = feature_encoder.transform(X)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    return X_features, y, feature_encoder, lb
