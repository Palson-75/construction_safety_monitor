from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "processed_ppe"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_NAMES = [
    "person", "helmet", "no_helmet", "vest", "no_vest",
    "boots", "no_boots", "gloves", "no_gloves", "goggles", "no_goggle"
]

def main():
    rows = []
    empty_rows = []
    split_summary = {}

    for split in ("train", "val", "test"):
        image_dir = DATASET / "images" / split
        label_dir = DATASET / "labels" / split
        images = [
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        labels = list(label_dir.glob("*.txt"))
        empty_count = 0
        invalid_count = 0
        for label in labels:
            text = label.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                empty_count += 1
                empty_rows.append([split, label.name])
                continue

            for line_no, line in enumerate(text.splitlines(), start=1):
                parts = line.split()
                reason = ""
                try:
                    if len(parts) != 5:
                        raise ValueError("wrong number of fields")
                    cls = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                    if not 0 <= cls < len(CLASS_NAMES):
                        reason = f"class_id_out_of_range: {cls}"
                    elif not all(0 <= v <= 1 for v in (x, y, w, h)):
                        reason = "coordinate_or_size_out_of_[0,1]"
                    elif w <= 0 or h <= 0:
                        reason = "zero_or_negative_width_height"
                    elif x - w / 2 < 0:
                        reason = f"left_edge_below_0: {x - w/2:.10f}"
                    elif y - h / 2 < 0:
                        reason = f"top_edge_below_0: {y - h/2:.10f}"
                    elif x + w / 2 > 1:
                        reason = f"right_edge_above_1: {x + w/2:.10f}"
                    elif y + h / 2 > 1:
                        reason = f"bottom_edge_above_1: {y + h/2:.10f}"
                except Exception as exc:
                    reason = f"parse_error: {exc}"

                if reason:
                    invalid_count += 1
                    rows.append([
                        split, label.name, line_no, line, reason
                    ])

        split_summary[split] = {
            "images": len(images),
            "labels": len(labels),
            "empty_labels": empty_count,
            "invalid_boxes": invalid_count,
        }

    out_invalid = DATASET / "invalid_box_details.csv"
    with out_invalid.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "label_file", "line_number", "line", "reason"])
        writer.writerows(rows)

    out_empty = DATASET / "empty_label_details.csv"
    with out_empty.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "label_file"])
        writer.writerows(empty_rows)

    print("SPLIT SUMMARY")
    for split, data in split_summary.items():
        print(f"\n{split.upper()}")
        for key, value in data.items():
            print(f"  {key}: {value}")

    print(f"\nInvalid boxes found: {len(rows)}")
    print(f"Details saved: {out_invalid}")
    print(f"Empty label files found: {len(empty_rows)}")
    print(f"Details saved: {out_empty}")
    print("\nNo files were modified.")

if __name__ == "__main__":
    main()
