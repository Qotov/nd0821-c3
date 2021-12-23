import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from starter.ml.model import train_model, compute_model_metrics, inference

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

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


@pytest.fixture(scope='session')
def data():
    X = np.array([[0, 0, 0, 0, 0, 0],
                  [3, 3, 3, 3, 3, 3],
                  [4, 4,4, 4, 4, 4],
                  [5, 5, 5, 5, 5, 5],
                  [6, 6, 6, 6, 6, 6]])

    y = np.array([1, 1, 1, 1, 0])
    return X, y


@pytest.fixture(scope='session')
def preds():
    return np.array([1, 1, 1, 1, 0])


def test_train_model(data):
    X_train, y_train = data

    model = LogisticRegression()
    model_ = train_model(X_train, y_train)
    assert type(model_) == type(model), \
        'model is not logreg, instead got type:{}'.format(type(model_))
    assert model_ is not model, \
        'object identity should be different'


def test_inference(data):
    X, y = data
    model = LogisticRegression()
    model = model.fit(X, y)
    preds = inference(model, X)

    assert isinstance(preds, np.ndarray), \
        'preds is not an array, instead got type: {}'.format(type(preds))
    assert len(preds) == len(y), \
        f'length of predicted values do not match, expected: {len(y)}, instead got: {len(preds)}'


def test_compute_model_metrics(data, preds):
    _, y = data
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert isinstance(precision, float), \
        'precision in not a float, instead got type: {}'.format(type(precision))
    assert isinstance(recall, float), \
        'recall in not a float, instead got type: {}'.format(type(recall))
    assert isinstance(fbeta, float), \
        'fbeta in not a float, instead got type: {}'.format(type(fbeta))