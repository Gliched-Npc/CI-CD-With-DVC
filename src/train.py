from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

os.makedirs("model", exist_ok=True)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

joblib.dump(train_df, "model/train_data.pkl")
joblib.dump(test_df, "model/test_data.pkl")

# Save a simple model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(train_df.drop("target", axis=1), train_df["target"])

joblib.dump(model, "model/iris_model.pkl")
