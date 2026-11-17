
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    ROOT / "outputs" / "full_video_ppe"
    / "combined_worker_dataset.csv"
)

df = pd.read_csv(CSV_PATH)

print("\n========== DATASET OVERVIEW ==========")
print("Total rows:", len(df))
print("Videos:", df["video_name"].nunique())
print("Video-worker pairs:",
      df[["video_name", "worker_id"]]
      .drop_duplicates().shape[0])

print("\n========== STATUS BY VIDEO ==========")
print(pd.crosstab(
    df["video_name"],
    df["final_status"]
).to_string())

print("\n========== STATUS BY PPE CATEGORY ==========")
print(pd.crosstab(
    df["ppe_category"],
    df["final_status"]
).to_string())

print("\n========== EVIDENCE SUMMARY ==========")
print(
    df.groupby("ppe_category")[
        ["positive_detections",
         "negative_detections",
         "total_detections"]
    ].sum().to_string()
)

print("\n========== WORKERS PER VIDEO ==========")
print(
    df.groupby("video_name")["worker_id"]
    .nunique().to_string()
)

print("\n========== UNKNOWN OBSERVATIONS ==========")
unknown = df[df["final_status"] == "UNKNOWN"]

print("Unknown rows:", len(unknown))
print(
    unknown.groupby("ppe_category").size()
    .sort_values(ascending=False).to_string()
)

print("\nAnalysis completed.")
