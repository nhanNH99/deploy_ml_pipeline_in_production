import os
import joblib
from starter.ml.model import inference, compute_model_metrics
from starter.ml.data import process_data
import pandas as pd

try:
    # Try to construct the path based on the "model" directory
    model_path = os.path.join(os.path.abspath("model"), "model.pkl")

    # Check if the file exists at the constructed path
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"File not found at path: {model_path}")

except FileNotFoundError:
    # If the file is not found, try a different path
    model_path = os.path.join(os.path.abspath("starter"), "model", "model.pkl")

    # Check if the file exists at the new path
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"File not found at path: {model_path}")
cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


def clean_data(data):
    """
    Clean the data by removing missing values and converting categorical values to numerical values.
    """
    dataframe = pd.DataFrame(data, index=[0])
    return dataframe


def main_inference(input_data: dict):
    """
    Main the inference function from the model module.
    """
    input_df = pd.DataFrame({k: v for k, v in input_data.items()}, index=[0])
    [model, encoder, lb] = joblib.load(model_path)
    x_test, _, _, _ = process_data(
        input_df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    # Call the inference function
    result = inference(model, x_test)
    if result[0] == 0:
        return "Salary <= 50K"
    else:
        return "Salary > 50K"


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


# if __name__ == "__main__":
