Student Performance Prediction (Machine Learning Project)

A machine learning web application that predicts a student's final academic performance (G3) based on study habits, academic history, and behavioural factors.

The project trains a machine learning model on a student dataset and deploys the prediction system through an interactive Streamlit web application.

Project Overview

Student academic performance is influenced by many factors such as study time, previous grades, attendance, and past academic failures. This project uses machine learning to model these relationships and predict the final student grade.

The application allows users to:

Input student academic information

Predict the student's final grade

View a confidence range for the prediction

Understand which features influence performance the most


Features

✔ End-to-end machine learning pipeline
✔ Data preprocessing and feature engineering
✔ Model training using Random Forest
✔ Prediction confidence range using tree distribution
✔ Interactive web application
✔ Feature importance visualization
✔ Modular project structure (separate prediction logic)


Machine Learning Model

The project uses:

Model: Random Forest Regressor

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

Prediction uncertainty is estimated using the distribution of predictions from individual trees in the forest.


Dataset

The dataset contains information about students including:

Age

Study time

Past academic failures

Absences

First period grade (G1)

Second period grade (G2)

Family background variables

Social and behavioural factors

The target variable is:

G3 — Final student grade

Grade range:

0 – 20



Project Structure
Student-Performance-Ml-pj
│
├── app
│   └── app.py                # Streamlit application
│
├── src
│   └── prediction.py         # Prediction logic
│
├── models
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── columns.pkl
│   └── metrics.pkl
│
├── data
│   └── student-mat.csv
│
├── notebooks
│   └── exploration.ipynb     # EDA and experimentation
│
├── requirements.txt
│
└── README.md
⚙️ Installation


Clone the repository:

git clone https://github.com/Iteoluwakishi/Student-Performance-Ml-pj.git

Navigate into the project directory:

cd Student-Performance-Ml-pj

Install dependencies:

pip install -r requirements.txt
▶️ Running the Application

Start the Streamlit app:

streamlit run app/app.py

The application will open in your browser.

📈 Example Prediction

Input example:

Age: 17
Study Time: 3
Past Failures: 0
Absences: 5
G1: 12
G2: 14

Output:

Predicted Final Grade: 13.8
Likely Grade Range: 12.4 – 15.2
Performance Level: Average


📊 Feature Importance

The application also visualizes the top factors influencing student performance, which helps explain how the model makes predictions.

Examples of important features:

Previous grades (G1, G2)

Study time

Absences

Past failures



🛠 Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Streamlit



Future Improvements

Possible enhancements:

SHAP explainability for model predictions

More student features in the input interface

Model comparison (XGBoost / Gradient Boosting)

Deployment to Streamlit Cloud

Real-time prediction API


Author

Adeniran Iteoluwakishi

Machine Learning | Data Science | Biomedical Data Science

GitHub:
https://github.com/Iteoluwakishi

📜 License

This project is open-source and available under the MIT License.
