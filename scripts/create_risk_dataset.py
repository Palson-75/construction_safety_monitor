
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "outputs" / "full_video_ppe"
    / "worker_features.csv"
)

OUTPUT_FILE = (
    ROOT / "outputs" / "full_video_ppe"
    / "final_risk_dataset.csv"
)

df = pd.read_csv(INPUT_FILE)

def assign_risk(row):

    helmet = row["helmet_status"]
    vest = row["vest_status"]

    # Reliable evidence of missing PPE
    if helmet == "NEGATIVE" or vest == "NEGATIVE":
        return "High"

    # Both essential PPE items detected
    if helmet == "DETECTED" and vest == "DETECTED":
        return "Low"

    # Not enough reliable evidence
    return "Insufficient Evidence"


df["risk_level"] = df.apply(assign_risk, axis=1)

df.to_csv(OUTPUT_FILE, index=False)

print("\n========== FINAL RISK DATASET ==========")

print("Total worker records:", len(df))

print("\nRisk distribution:")
print(df["risk_level"].value_counts())

print("\nRisk percentages:")
print(
    (df["risk_level"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nVideo-wise risk distribution:")
print(
    pd.crosstab(df["video_name"], df["risk_level"])
)

print("\nSaved to:", OUTPUT_FILE)
