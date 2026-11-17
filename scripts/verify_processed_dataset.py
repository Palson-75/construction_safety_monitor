from pathlib import Path
from collections import Counter
import csv

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "processed_ppe"
CLASSES = [
    "person", "helmet", "no_helmet", "vest", "no_vest",
    "boots", "no_boots", "gloves", "no_gloves", "goggles", "no_goggle"
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def main():
    if not DATASET.exists():
        raise SystemExit(f"Dataset not found: {DATASET}")

    total_images = total_labels = empty_labels = missing_labels = 0
    invalid_boxes = 0
    overall = Counter()
    split_counts = {}

    for split in ("train", "val", "test"):
        image_dir = DATASET / "images" / split
        label_dir = DATASET / "labels" / split

        images = [
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ] if image_dir.exists() else []
        labels = list(label_dir.glob("*.txt")) if label_dir.exists() else []
        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}

        missing = image_stems - label_stems
        orphan = label_stems - image_stems
        counts = Counter()
        split_empty = 0
        split_invalid = 0

        for label in labels:
            content = label.read_text(encoding="utf-8", errors="replace").strip()
            if not content:
                split_empty += 1
                continue

            for line in content.splitlines():
                parts = line.split()
                if len(parts) != 5:
                    split_invalid += 1
                    continue
                try:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                except ValueError:
                    split_invalid += 1
                    continue

                if not (
                    0 <= class_id < len(CLASSES)
                    and 0 <= x <= 1 and 0 <= y <= 1
                    and 0 < w <= 1 and 0 < h <= 1
                    and x - w / 2 >= 0 and y - h / 2 >= 0
                    and x + w / 2 <= 1 and y + h / 2 <= 1
                ):
                    split_invalid += 1
                    continue

                counts[class_id] += 1
                overall[class_id] += 1

        total_images += len(images)
        total_labels += len(labels)
        empty_labels += split_empty
        missing_labels += len(missing)
        invalid_boxes += split_invalid

        split_counts[split] = {
            "images": len(images),
            "label_files": len(labels),
            "empty_labels": split_empty,
            "images_missing_labels": len(missing),
            "orphan_labels": len(orphan),
            "invalid_boxes": split_invalid,
            "objects": sum(counts.values()),
        }

    print("\nPROCESSED DATASET VERIFICATION")
    print("=" * 38)
    for split, data in split_counts.items():
        print(f"\n{split.upper()}")
        for key, value in data.items():
            print(f"  {key}: {value}")

    print("\nCLASS DISTRIBUTION (OBJECTS)")
    print("-" * 38)
    for class_id, name in enumerate(CLASSES):
        print(f"{class_id:2d} {name:12s}: {overall[class_id]}")

    print("\nTOTALS")
    print("-" * 38)
    print(f"Images: {total_images}")
    print(f"Label files: {total_labels}")
    print(f"Empty label files: {empty_labels}")
    print(f"Images missing labels: {missing_labels}")
    print(f"Invalid boxes remaining: {invalid_boxes}")

    report = DATASET / "verification_report.csv"
    with report.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "metric", "value"])
        for split, data in split_counts.items():
            for metric, value in data.items():
                writer.writerow([split, metric, value])
        for class_id, name in enumerate(CLASSES):
            writer.writerow(["all", f"class_{class_id}_{name}", overall[class_id]])

    print(f"\nSaved report: {report}")

if __name__ == "__main__":
    main()
