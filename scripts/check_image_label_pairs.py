
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASETS = {
    "Ultralytics": PROJECT_DIR / "datasets" / "construction_ppe",
    "Mendeley": PROJECT_DIR / "datasets" / "mendeley_ppe",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

for name, root in DATASETS.items():
    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    total_images = 0
    total_labels = 0

    for label_dir in root.rglob("labels"):
        image_dir = label_dir.parent / "images"

        if not image_dir.exists():
            print("No matching images folder for:", label_dir)
            continue

        labels = {
            p.relative_to(label_dir).with_suffix("")
            for p in label_dir.rglob("*.txt")
        }

        images = {
            p.relative_to(image_dir).with_suffix("")
            for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        }

        missing_images = labels - images
        missing_labels = images - labels

        total_images += len(images)
        total_labels += len(labels)

        print("\nSplit:", label_dir.parent.name)
        print("Images:", len(images))
        print("Labels:", len(labels))
        print("Labels without images:", len(missing_images))
        print("Images without labels:", len(missing_labels))

        if missing_images:
            print("Unmatched label examples:")
            for item in sorted(missing_images)[:10]:
                print(" ", item)

        if missing_labels:
            print("Unmatched image examples:")
            for item in sorted(missing_labels)[:10]:
                print(" ", item)

    print("\nTOTAL IMAGES:", total_images)
    print("TOTAL LABELS:", total_labels)
