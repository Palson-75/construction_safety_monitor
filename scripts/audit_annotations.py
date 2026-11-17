
from pathlib import Path
from collections import defaultdict
import cv2
import csv
import random

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs" / "annotation_audit"
OUTPUT.mkdir(parents=True, exist_ok=True)

random.seed(42)

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

DATASETS = [
    {
        "name": "Ultralytics",
        "root": ROOT / "datasets" / "construction_ppe",
        "splits": ["train", "val", "test"],
        "layout": "ultralytics",
        "classes": {
            0: "helmet",
            1: "gloves",
            2: "vest",
            3: "boots",
            4: "goggles",
            5: "none",
            6: "Person",
            7: "no_helmet",
            8: "no_goggle",
            9: "no_gloves",
            10: "no_boots",
        },
    },
    {
        "name": "Mendeley",
        "root": ROOT / "datasets" / "mendeley_ppe",
        "splits": ["train", "valid"],
        "layout": "mendeley",
        "classes": {
            0: "Helmet",
            1: "NoHelmet",
            2: "NoVest",
            3: "Vest",
        },
    },
]

MAX_EXAMPLES = 20
CELL_WIDTH = 360
CELL_HEIGHT = 280
COLUMNS = 4


def find_image(image_dir, stem):
    for ext in IMAGE_EXTENSIONS:
        path = image_dir / (stem + ext)
        if path.exists():
            return path
    return None


def make_contact_sheet(examples, output_path, class_name):
    if not examples:
        return

    examples = examples[:MAX_EXAMPLES]
    rows = (len(examples) + COLUMNS - 1) // COLUMNS

    sheet = 255 * __import__("numpy").ones(
        (rows * CELL_HEIGHT, COLUMNS * CELL_WIDTH, 3),
        dtype="uint8"
    )

    for index, (image_path, boxes) in enumerate(examples):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        h, w = image.shape[:2]
        scale = min(340 / w, 235 / h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        image = cv2.resize(image, (new_w, new_h))

        for x, y, bw, bh in boxes:
            x1 = int((x - bw / 2) * new_w)
            y1 = int((y - bh / 2) * new_h)
            x2 = int((x + bw / 2) * new_w)
            y2 = int((y + bh / 2) * new_h)

            cv2.rectangle(
                image, (x1, y1), (x2, y2),
                (0, 200, 0), 2
            )

        row = index // COLUMNS
        col = index % COLUMNS

        x_offset = col * CELL_WIDTH + 10
        y_offset = row * CELL_HEIGHT + 30

        sheet[
            y_offset:y_offset + new_h,
            x_offset:x_offset + new_w
        ] = image

        filename = image_path.name[:35]
        cv2.putText(
            sheet,
            filename,
            (x_offset, row * CELL_HEIGHT + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1
        )

    cv2.imwrite(str(output_path), sheet)


report_rows = []

for dataset in DATASETS:
    print(f"\n{'=' * 55}")
    print(f"DATASET: {dataset['name']}")
    print("=" * 55)

    class_stats = defaultdict(lambda: {
        "objects": 0,
        "images": set(),
        "invalid": 0,
        "areas": [],
        "examples": [],
    })

    for split in dataset["splits"]:

        if dataset["layout"] == "ultralytics":
            label_dir = dataset["root"] / "labels" / split
            image_dir = dataset["root"] / "images" / split
        else:
            label_dir = dataset["root"] / split / "labels"
            image_dir = dataset["root"] / split / "images"

        if not label_dir.exists():
            print(f"Missing label directory: {label_dir}")
            continue

        if not image_dir.exists():
            print(f"Missing image directory: {image_dir}")
            continue

        for label_file in label_dir.glob("*.txt"):
            image_path = find_image(image_dir, label_file.stem)

            if image_path is None:
                continue

            parsed = defaultdict(list)

            try:
                lines = label_file.read_text().splitlines()
            except Exception:
                continue

            for line in lines:
                parts = line.split()

                if len(parts) != 5:
                    continue

                try:
                    class_id = int(float(parts[0]))
                    x, y, bw, bh = map(float, parts[1:])
                except ValueError:
                    continue

                if class_id not in dataset["classes"]:
                    continue

                stats = class_stats[class_id]
                stats["objects"] += 1
                stats["images"].add(str(image_path))

                valid = (
                    0 <= x <= 1
                    and 0 <= y <= 1
                    and 0 < bw <= 1
                    and 0 < bh <= 1
                    and x - bw / 2 >= 0
                    and y - bh / 2 >= 0
                    and x + bw / 2 <= 1
                    and y + bh / 2 <= 1
                )

                if not valid:
                    stats["invalid"] += 1

                stats["areas"].append(bw * bh)
                parsed[class_id].append((x, y, bw, bh))

            for class_id, boxes in parsed.items():
                stats = class_stats[class_id]

                if len(stats["examples"]) < MAX_EXAMPLES * 3:
                    stats["examples"].append((image_path, boxes))

    dataset_output = OUTPUT / dataset["name"]
    dataset_output.mkdir(parents=True, exist_ok=True)

    for class_id, class_name in dataset["classes"].items():
        stats = class_stats[class_id]

        areas = stats["areas"]
        avg_area = sum(areas) / len(areas) if areas else 0

        print(
            f"{class_name:15} "
            f"Objects: {stats['objects']:5} | "
            f"Images: {len(stats['images']):5} | "
            f"Invalid boxes: {stats['invalid']:4} | "
            f"Avg box area: {avg_area:.4f}"
        )

        report_rows.append({
            "dataset": dataset["name"],
            "class_id": class_id,
            "class_name": class_name,
            "objects": stats["objects"],
            "images": len(stats["images"]),
            "invalid_boxes": stats["invalid"],
            "average_box_area": round(avg_area, 6),
        })

        random.shuffle(stats["examples"])

        make_contact_sheet(
            stats["examples"],
            dataset_output / f"{class_id}_{class_name}.jpg",
            class_name
        )

report_path = OUTPUT / "annotation_report.csv"

with open(report_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=report_rows[0].keys())
    writer.writeheader()
    writer.writerows(report_rows)

print("\nAudit completed!")
print(f"CSV report: {report_path}")
print(f"Contact sheets: {OUTPUT}")
