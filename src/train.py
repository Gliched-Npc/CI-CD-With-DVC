from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
import os

# Load iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

# Split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create model folder
os.makedirs("model", exist_ok=True)

# Save train & test
joblib.dump(train_df, "model/train_data.pkl")
joblib.dump(test_df, "model/test_data.pkl")

# Train model
model = RandomForestClassifier()
model.fit(train_df.drop("target", axis=1), train_df["target"])

# Save model
joblib.dump(model, "model/iris_model.pkl")

print("Training complete.")
