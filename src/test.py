import joblib
from sklearn.metrics import accuracy_score

# Load model and test data
model = joblib.load("model/iris_model.pkl")
test_df = joblib.load("model/test_data.pkl")

X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc}")
