import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from preprocess import load_data, preprocess_data, split_data, scale_data


df = load_data(r"C:\Users\USER\OneDrive\Documents\Student-Performance-Ml-pj\data\student-mat.csv")

X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)
X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, y_train)
preds = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("MSE:", mse)
print("R2 Score:", r2)


joblib.dump(model, "C:\\Users\\USER\\OneDrive\\Documents\\Student-Performance-Ml-pj\\models\\model.pkl")
joblib.dump(scaler, "C:\\Users\\USER\\OneDrive\\Documents\\Student-Performance-Ml-pj\\models\\scaler.pkl")