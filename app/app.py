import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import predict_student

@st.cache_resource
def load_model():
    return joblib.load("models/model.pkl")

@st.cache_resource
def load_columns():
    return joblib.load("models/columns.pkl")

@st.cache_resource
def load_metrics():
    return joblib.load("models/metrics.pkl")


model = load_model()
columns = load_columns()
metrics = load_metrics()


st.title("🎓 Student Performance Predictor")

st.write("This machine learning model predicts a student's final grade based on academic and behavioral factors.")


st.subheader("Model Performance")

st.write(f"R² Score: {metrics['r2']:.2f}")
st.write(f"Mean Squared Error: {metrics['mse']:.2f}")
st.subheader("Enter Student Information")

age = st.slider("Age", 15, 22)
studytime = st.slider("Study Time (1 = <2 hrs, 4 = >10 hrs)", 1, 4)
failures = st.slider("Past Class Failures", 0, 4)
absences = st.slider("Number of School Absences", 0, 100)
G1 = st.slider("First Period Grade (G1)", 0, 20)
G2 = st.slider("Second Period Grade (G2)", 0, 20)

if st.button("Predict Final Grade"):

    input_dict = {"age": age, "studytime": studytime, "failures": failures, "absences": absences, "G1": G1, "G2": G2}

    input_df = pd.DataFrame([input_dict])

    prediction, lower, upper = predict_student(input_df)

    st.success(f"Predicted Final Grade: {prediction:.2f} / 20")
    st.info(f"Likely Grade Range: {lower:.2f} – {upper:.2f}")

    if prediction >= 15:
        st.success("Performance Level: Excellent")
    elif prediction >= 10:
        st.warning("Performance Level: Average")
    else:
        st.error("Performance Level: At Risk")


st.subheader("Top Factors Affecting Student Performance")

importances = model.feature_importances_

feature_importance = pd.DataFrame({"feature": columns, "importance": importances})

feature_importance = feature_importance.sort_values(by="importance", ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8, 5))

ax.barh(feature_importance["feature"], feature_importance["importance"])
ax.set_xlabel("Importance Score")
ax.set_title("Top 10 Features Influencing Student Performance")
ax.invert_yaxis()

st.pyplot(fig)