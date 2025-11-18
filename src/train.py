from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

# Load iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

# Create model directory
os.makedirs("model", exist_ok=True)

# Split into train and test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Prepare train and test sets
X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]

X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]

# Save train and test data as tuples
joblib.dump((X_train, y_train), "model/train_data.pkl")
joblib.dump((X_test, y_test), "model/test_data.pkl")

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/iris_model.pkl")

print("Training complete. Model and data saved.")
