import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

df = pd.read_csv("data/Iris.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/iris_model.pkl")
joblib.dump((X_train, y_train), "model/train_data.pkl")
joblib.dump((X_test, y_test), "model/test_data.pkl")

print("Training completed.")
