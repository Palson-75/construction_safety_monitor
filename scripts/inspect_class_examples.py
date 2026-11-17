from pathlib import Path
import cv2
import random

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs" / "class_inspection"
OUTPUT.mkdir(parents=True, exist_ok=True)

random.seed(42)

DATASETS = [
    {
        "name": "Ultralytics",
        "root": ROOT / "datasets" / "construction_ppe",
        "splits": ["train", "val", "test"],
        "layout": "ultralytics",
        "classes": {
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

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

for dataset in DATASETS:
    print(f"\nInspecting {dataset['name']}...")

    for class_id, class_name in dataset["classes"].items():
        examples = []

        for split in dataset["splits"]:
            if dataset["layout"] == "ultralytics":
               label_dir = dataset["root"] / "labels" / split
               image_dir = dataset["root"] / "images" / split
            else:
               label_dir = dataset["root"] / split / "labels"
               image_dir = dataset["root"] / split / "images"

            if not label_dir.exists() or not image_dir.exists():
                continue

            for label_file in label_dir.glob("*.txt"):
                try:
                    lines = label_file.read_text().splitlines()
                except Exception:
                    continue

                matching_boxes = []

                for line in lines:
                    parts = line.split()

                    if len(parts) != 5:
                        continue

                    try:
                        cid = int(float(parts[0]))
                        if cid == class_id:
                            matching_boxes.append(
                                [float(x) for x in parts[1:]]
                            )
                    except ValueError:
                        continue

                if not matching_boxes:
                    continue

                image_path = next(
                    (
                        image_dir / (label_file.stem + ext)
                        for ext in IMAGE_EXTENSIONS
                        if (image_dir / (label_file.stem + ext)).exists()
                    ),
                    None,
                )

                if image_path:
                    examples.append((image_path, matching_boxes))

        random.shuffle(examples)

        for index, (image_path, boxes) in enumerate(examples[:5]):
            image = cv2.imread(str(image_path))

            if image is None:
                continue

            height, width = image.shape[:2]

            for x, y, w, h in boxes:
                x1 = int((x - w / 2) * width)
                y1 = int((y - h / 2) * height)
                x2 = int((x + w / 2) * width)
                y2 = int((y + h / 2) * height)

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    class_name,
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            filename = f"{dataset['name']}_{class_name}_{index + 1}.jpg"
            cv2.imwrite(str(OUTPUT / filename), image)

        print(
            f"{class_name}: found {len(examples)} images; "
            f"saved {min(5, len(examples))} examples"
        )

print(f"\nFinished! Images saved in:\n{OUTPUT}")