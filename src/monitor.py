import joblib
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Load saved data
train_df = joblib.load("model/train_data.pkl")
test_df = joblib.load("model/test_data.pkl")

# Create data drift report
report = Report(metrics=[DataDriftPreset()])

report.run(
    reference_data=train_df,
    current_data=test_df
)

report.save_html("model/data_drift_report.html")

print("Data drift report saved.")
