import joblib
import numpy as np
import pandas as pd

# Load saved artifacts
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
columns = joblib.load("models/columns.pkl")


def predict_student(data: pd.DataFrame):
    
    data = data.reindex(columns=columns, fill_value=0)

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]

    tree_predictions = [tree.predict(data_scaled)[0] for tree in model.estimators_]

    lower = np.percentile(tree_predictions, 10)
    upper = np.percentile(tree_predictions, 90)

    return prediction, lower, upper