import pandas as pd
import joblib
from evidently.report import Report
from evidently.metrics import DataDriftTable

train_df = joblib.load("model/train_data.pkl")
test_df = joblib.load("model/test_data.pkl")

# reset index (important)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Remove empty columns if any
train_df = train_df.dropna(axis=1, how="all")
test_df = test_df.dropna(axis=1, how="all")

# Ensure columns match
common_cols = list(set(train_df.columns) & set(test_df.columns))
train_df = train_df[common_cols]
test_df = test_df[common_cols]

report = Report(metrics=[DataDriftTable()])
report.run(reference_data=train_df, current_data=test_df)

report.save_html("model/data_drift_report.html")
