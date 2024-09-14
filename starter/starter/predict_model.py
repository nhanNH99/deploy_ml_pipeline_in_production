import os

import joblib
import pandas as pd

from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference

try:
    model_path = os.path.join(os.path.abspath("model"), "model.pkl")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"File not found at path: {model_path}")
except FileNotFoundError:
    model_path = os.path.join(os.path.abspath("starter"), "model", "model.pkl")
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
