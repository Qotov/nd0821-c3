# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- LogReg model with default hyperparameters.
- model version: 1.0.0
- model date: 23 Dec 2021

## Intended Use
- This model is intended to predict the Census Income data (>50K or <=50K) based on the list of different features.
## Training Data
- The data is from UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income).
- The dataset contains 48842 rows, and after an 80-20 split 80% of the data was used for training.
## Evaluation Data
- 20% of the dataset was used for testing purposes.

## Metrics
- Evaluation metrics include precision (0.67), recall (0.32) and fbeta (0.43).

## Ethical Considerations
- Demographic data were obtained from the public 1994 Census Database. No new information is inferred or annotated.

## Caveats and Recommendations
- Hyper-parameter tuning.
- Try different model
