
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "full_video_ppe"

REPORT_FILES = {
    "video1": "worker_ppe_report_vid1.csv",
    "video2": "worker_ppe_report_new.csv",
    **{
        f"video{i}": f"worker_ppe_report_video{i}.csv"
        for i in range(3, 11)
    }
}

all_reports = []

for video_name, filename in REPORT_FILES.items():
    file_path = REPORT_DIR / filename

    if not file_path.exists():
        print(f"Missing report: {filename}")
        continue

    df = pd.read_csv(file_path)
    df["video_name"] = video_name
    all_reports.append(df)

if not all_reports:
    raise FileNotFoundError("No worker reports were found.")

combined = pd.concat(all_reports, ignore_index=True)

columns = [
    "video_name",
    "worker_id",
    "frames_seen",
    "ppe_category",
    "positive_detections",
    "negative_detections",
    "total_detections",
    "positive_ratio",
    "final_status"
]

combined = combined[columns]

output_path = REPORT_DIR / "combined_worker_dataset.csv"
combined.to_csv(output_path, index=False)

print("\nReports combined successfully!")
print("Videos included:", combined["video_name"].nunique())
print("Total rows:", len(combined))
print(
    "Unique video-worker pairs:",
    combined[["video_name", "worker_id"]]
    .drop_duplicates()
    .shape[0]
)

print("\nRows per video:")
print(combined.groupby("video_name").size())

print("\nStatus distribution:")
print(combined["final_status"].value_counts())

print("\nSaved to:", output_path)
