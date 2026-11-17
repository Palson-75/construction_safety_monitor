
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "datasets" / "processed_ppe" / "labels" / "train"

TARGET_CLASSES = {
    6: "no_boots",
    8: "no_gloves",
    10: "no_goggle"
}

random.seed(42)

for class_id, class_name in TARGET_CLASSES.items():
    matching_files = []

    for label_file in LABEL_DIR.glob("*.txt"):
        for line in label_file.read_text().splitlines():
            parts = line.split()

            if len(parts) == 5 and int(parts[0]) == class_id:
                matching_files.append(label_file)
                break

    matching_files = list(set(matching_files))
    random.shuffle(matching_files)

    print(f"\n{class_name}: {len(matching_files)} images")
    for file in matching_files[:10]:
        print(file.name)
