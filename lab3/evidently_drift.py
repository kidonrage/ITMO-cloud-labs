from sklearn.datasets import load_iris

from evidently import Report
from evidently.presets import DataDriftPreset

def main():
    iris = load_iris(as_frame=True)
    data = iris.frame
    
    ref = data.sample(frac=0.7, random_state=42)
    cur = data.drop(ref.index)

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref, current_data=cur)

    snapshot.save_html("evidently_drift_report.html")
    print("Saved: evidently_drift_report.html")

if __name__ == "__main__":
    main()
