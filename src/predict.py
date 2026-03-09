import joblib
import numpy as np

model = joblib.load("C:\\Users\\USER\\OneDrive\\Documents\\Student-Performance-Ml-pj\\models\\model.pkl")
scaler = joblib.load("C:\\Users\\USER\\OneDrive\\Documents\\Student-Performance-Ml-pj\\models\\scaler.pkl")

def predict_student(data):

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    return prediction[0]