import joblib
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping

train_df = joblib.load("model/train_data.pkl")
test_df = joblib.load("model/test_data.pkl")

# Tell Evidently not to search for datetime column
column_mapping = ColumnMapping()
column_mapping.datetime = None
column_mapping.target = "target"
column_mapping.numerical_features = [col for col in train_df.columns if col != "target"]
column_mapping.categorical_features = []

report = Report(metrics=[DataDriftPreset()])
report.run(
    reference_data=train_df,
    current_data=test_df,
    column_mapping=column_mapping
)

report.save_html("model/data_drift_report.html")

print("Drift report generated.")
