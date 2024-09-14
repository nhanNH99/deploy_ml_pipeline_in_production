# Script to train machine learning model.
import json
import os

import joblib
import pandas as pd
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    compute_slice_metrics,
    inference,
    train_model,
)
from sklearn.model_selection import KFold

# Add code to load in the data.
data_path = os.path.join(os.path.abspath("data"), "census.csv")
data = pd.read_csv(
    "D:/03.FPTSoft/devops/Session3/deploy_ml_pipeline_in_production/starter/data/census.csv"
)

data.columns = data.columns.str.strip()

# Optional enhancement, use K-fold cross validation instead of a train-test split.
kf = KFold(n_splits=5, random_state=42, shuffle=True)

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
precision_scores = []
recall_scores = []
fbeta_scores = []
dict_output = []
# Perform K-Fold Cross-Validation
for fold, (train_index, test_index) in enumerate(kf.split(data), 1):
    train = data.iloc[train_index]
    test = data.iloc[test_index]

    # Process the training and test data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Train the model
    model = train_model(X_train, y_train)
    with open("model.pkl", "wb") as f:
        joblib.dump([model, encoder, lb], "model.pkl")
    # Make predictions
    preds = inference(model, X_test)
    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    precision_scores.append(precision)
    recall_scores.append(recall)
    fbeta_scores.append(fbeta)
    for cat_feature in cat_features:
        for category in data[cat_feature].unique():
            dict_output.append(
                {
                    "fold": fold,
                    "precision": precision,
                    "recall": recall,
                    "fbeta": fbeta,
                    "feature": cat_feature,
                    "category": category,
                }
            )
dict_output_str = json.dumps(dict_output, indent=4)
with open("slice_output.txt", "w") as f:
    f.write(dict_output_str)

avg_precision = sum(precision_scores) / len(precision_scores)
avg_recall = sum(recall_scores) / len(recall_scores)
avg_fbeta = sum(fbeta_scores) / len(fbeta_scores)

print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average F-beta: {avg_fbeta:.4f}")
