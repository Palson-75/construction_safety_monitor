
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "outputs" / "test_evaluation" / "confusion_matrix.csv"

if not CSV_PATH.exists():
    print("CSV not found. We will use the PNG plots instead.")
else:
    df = pd.read_csv(CSV_PATH, index_col=0)
    print("\nCONFUSION MATRIX")
    print(df.to_string())
