
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FILE = (
    ROOT / "outputs" / "full_video_ppe"
    / "worker_features.csv"
)

df = pd.read_csv(FILE)

categories = ["helmet", "vest", "boots", "gloves", "goggles"]

print("\n========== STATUS COUNTS ==========")

for category in categories:
    print(f"\n{category.upper()}")
    print(df[f"{category}_status"].value_counts())

print("\n========== WORKER STATUS SUMMARY ==========")

for category in categories:
    print(
        category,
        "DETECTED:", (df[f"{category}_status"] == "DETECTED").sum(),
        "NEGATIVE:", (df[f"{category}_status"] == "NEGATIVE").sum(),
        "UNKNOWN:", (df[f"{category}_status"] == "UNKNOWN").sum(),
        "CONFLICT:", (df[f"{category}_status"] == "CONFLICT").sum()
    )

# Count workers with at least one reliable negative observation
negative_counts = {}

for category in categories:
    negative_counts[category] = (
        df[f"{category}_status"] == "NEGATIVE"
    ).sum()

print("\n========== RELIABLE NEGATIVE EVIDENCE ==========")
print(negative_counts)

print("\nWorkers with at least one NEGATIVE status:")
print(
    df[
        df[[f"{c}_status" for c in categories]]
        .eq("NEGATIVE").any(axis=1)
    ][["video_name", "worker_id"]].shape[0]
)

print("\nWorkers with all PPE categories UNKNOWN:")
print(
    df[
        df[[f"{c}_status" for c in categories]]
        .eq("UNKNOWN").all(axis=1)
    ].shape[0]
)
