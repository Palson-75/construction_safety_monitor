
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "outputs" / "full_video_ppe"
    / "combined_worker_dataset.csv"
)

OUTPUT_FILE = (
    ROOT / "outputs" / "full_video_ppe"
    / "worker_features.csv"
)

df = pd.read_csv(INPUT_FILE)

# One record per video-worker pair
features = (
    df.groupby(["video_name", "worker_id"], as_index=False)
    ["frames_seen"].max()
)

categories = [
    "helmet",
    "vest",
    "boots",
    "gloves",
    "goggles"
]

for category in categories:
    subset = df[df["ppe_category"] == category].copy()

    subset = subset[
        [
            "video_name",
            "worker_id",
            "positive_detections",
            "negative_detections",
            "total_detections",
            "positive_ratio",
            "final_status"
        ]
    ]

    subset = subset.rename(columns={
        "positive_detections": f"{category}_positive",
        "negative_detections": f"{category}_negative",
        "total_detections": f"{category}_total",
        "positive_ratio": f"{category}_positive_ratio",
        "final_status": f"{category}_status"
    })

    features = features.merge(
        subset,
        on=["video_name", "worker_id"],
        how="left"
    )

# Preserve missing evidence as missing, not zero
numeric_columns = [
    column for column in features.columns
    if column.endswith(("_positive", "_negative", "_total",
                        "_positive_ratio"))
]

features[numeric_columns] = features[numeric_columns].fillna(0)

# Save the feature dataset
features.to_csv(OUTPUT_FILE, index=False)

print("\n========== WORKER FEATURE DATASET ==========")
print("Total workers:", len(features))
print("Total columns:", len(features.columns))

print("\nDataset preview:")
print(features.head(10).to_string(index=False))

print("\nMissing status counts:")
for category in categories:
    print(
        category,
        features[f"{category}_status"].isna().sum()
    )

print("\nSaved to:", OUTPUT_FILE)
