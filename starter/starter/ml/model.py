import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV


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
    model_random_forest
        Trained machine learning model (Random Forest).
    """

    # Implement hyperparameter tuning here
    model_random_forest = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, 30, 40, 50],
        "min_samples_split": [2, 5, 10],
    }
    grid = GridSearchCV(model_random_forest, param_grid, cv=3, n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)
    model_random_forest = grid.best_estimator_
    return model_random_forest


def compute_slice_metrics(model, X, y, features):
    """
    Compute performance metrics for slices of data where the value of a given feature is fixed.

    Parameters:
    - model: Trained machine learning model.
    - X: Feature DataFrame.
    - y: Target Series.
    - feature: The feature on which to slice the data.

    Outputs metrics to 'slice_output.txt'.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    for feature in features:
        unique_values = X[feature].unique()
        metrics_output = []
        for value in unique_values:
            # Filter the data for the current slice
            slice_idx = X[feature] == value
            X_slice = X[slice_idx]
            y_slice = y[slice_idx]

            # Make predictions
            y_pred = model.predict(X_slice)

            # Compute metrics for the slice
            precision, recall, fbeta = compute_model_metrics(y_slice, y_pred)

            # Store the metrics
            metrics_output.append(
                f"Metrics for {feature} = {value}:\n"
                f"  Precision: {precision:.4f}\n"
                f"  Recall: {recall:.4f}\n"
                f"  F-beta Score: {fbeta:.4f}\n\n"
            )

    with open("slice_output.txt", "w") as f:
        f.writelines(metrics_output)


def compute_model_metrics(y, preds):
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
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : pkl file
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    results = model.predict(X)
    return results
