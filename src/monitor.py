import joblib
import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

X_train, _ = joblib.load("model/train_data.pkl")
X_test, _ = joblib.load("model/test_data.pkl")

columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X_train = pd.DataFrame(X_train, columns=columns)
X_test  = pd.DataFrame(X_test, columns=columns)

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=X_train, current_data=X_test)

os.makedirs("model", exist_ok=True)
report.save_html("model/data_drift_report.html")

print("Drift report created.")
