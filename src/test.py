import joblib
from sklearn.metrics import accuracy_score

model = joblib.load("model/iris_model.pkl")
X_test, y_test = joblib.load("model/test_data.pkl")

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {acc}")
