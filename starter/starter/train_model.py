# Script to train machine learning model.

from sklearn.model_selection import train_test_split

import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, save_model
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def data_slicing(feature, y_test, preds):
    data = np.column_stack((feature, y_test, preds))

    for cls in np.unique(feature):
        filter = np.asarray([cls])
        filtered = data[np.in1d(data[:, 0], filter)]
        precision, recall, fbeta = compute_model_metrics(filtered[:, 1].astype(int), filtered[:, 2].astype(int))

        logger.info(f"For {cls} - precision: {str(precision)}, recall {str(recall)}, f_beta {str(fbeta)}")

        with open('slice_out.txt', 'a') as file:
            file.write(f"For {cls} - precision: {str(precision)}, recall {str(recall)}, f_beta {str(fbeta)}\n")


# Add code to load in the data.

# Read cleaned version!
data = pd.read_csv("./data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(df, test_size=0.20)
feature = test["occupation"].to_numpy()

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

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, test_encoder, test_lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

save_model(encoder, "encoder.pkl")
save_model(lb, "label_binarizer.pkl")

# Train and save a model.
model = train_model(X_train, y_train)
save_model(model, 'model.pkl')
preds = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, preds)
logger.info(f"precision: {str(precision)}, recall {str(recall)}, f_beta {str(fbeta)}")

data_slicing(feature, y_test, preds)

