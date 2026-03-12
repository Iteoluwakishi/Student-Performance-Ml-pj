import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

columns = joblib.load("C:\\Users\\USER\\OneDrive\\Documents\\Student-Performance-Ml-pj\\models\\columns.pkl")
model = joblib.load("C:\\Users\\USER\\OneDrive\\Documents\\Student-Performance-Ml-pj\\models\\model.pkl")
scaler = joblib.load("C:\\Users\\USER\\OneDrive\\Documents\\Student-Performance-Ml-pj\\models\\scaler.pkl")
metrics = joblib.load("C:\\Users\\USER\\OneDrive\\Documents\\Student-Performance-Ml-pj\\models\\metrics.pkl")


st.title("🎓 Student Performance Predictor")

st.write(
"This machine learning model predicts a student's final grade based on academic and behavioral factors."
)


st.subheader("Model Performance")

st.write(f"R² Score: {metrics['r2']:.2f}")
st.write(f"Mean Squared Error: {metrics['mse']:.2f}")


st.subheader("Enter Student Information")

age = st.slider("Age", 15, 22)
studytime = st.slider("Study Time (1–4)", 1, 4)
failures = st.slider("Past Failures", 0, 4)
absences = st.slider("Absences", 0, 100)
G1 = st.slider("First Period Grade (G1)", 0, 20)
G2 = st.slider("Second Period Grade (G2)", 0, 20)


if st.button("Predict Final Grade"):

    input_dict = {
        "age": age,
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "G1": G1,
        "G2": G2,
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=columns, fill_value=0)

    input_scaled = scaler.transform(input_df)

    # Main prediction
    prediction = model.predict(input_scaled)[0]

    # Predictions from all trees (for uncertainty)
    tree_predictions = [tree.predict(input_scaled)[0] for tree in model.estimators_]

    std_dev = np.std(tree_predictions)

    lower = max(0, np.percentile(tree_predictions, 10))
    upper = min(20, np.percentile(tree_predictions, 90))

    st.success(f"Predicted Final Grade: {prediction:.2f} / 20")

    st.info(f"Confidence Range: {lower:.2f} – {upper:.2f}")

    if prediction >= 15:
        st.success("Performance: Excellent")
    elif prediction >= 10:
        st.warning("Performance: Average")
    else:
        st.error("Performance: At Risk")


st.subheader("Top Factors Affecting Student Performance")

importances = model.feature_importances_

feature_importance = pd.DataFrame({
    "feature": columns,
    "importance": importances
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
).head(10)

fig, ax = plt.subplots()

ax.barh(feature_importance["feature"], feature_importance["importance"])
ax.set_title("Top 10 Important Features")

st.pyplot(fig)