
from pathlib import Path
from collections import Counter

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASETS = {
    "Ultralytics": PROJECT_DIR / "datasets" / "construction_ppe",
    "Mendeley": PROJECT_DIR / "datasets" / "mendeley_ppe",
}

for name, root in DATASETS.items():
    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    if not root.exists():
        print("Dataset not found:", root)
        continue

    # Find all directories named labels, including nested ones
    label_dirs = [
        p for p in root.rglob("*")
        if p.is_dir() and p.name.lower() == "labels"
    ]

    label_files = []
    for folder in label_dirs:
        label_files.extend(folder.rglob("*.txt"))

    # Find image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    ]

    class_counts = Counter()
    empty_files = 0
    invalid_lines = 0

    for file in label_files:
        try:
            lines = file.read_text(
                encoding="utf-8"
            ).splitlines()

            if not any(line.strip() for line in lines):
                empty_files += 1

            for line in lines:
                if not line.strip():
                    continue

                parts = line.split()

                if len(parts) != 5:
                    invalid_lines += 1
                    continue

                try:
                    class_id = int(parts[0])
                    coordinates = [float(x) for x in parts[1:]]

                    if class_id < 0:
                        raise ValueError("Negative class ID")

                    if not all(0 <= x <= 1 for x in coordinates):
                        raise ValueError("Invalid coordinates")

                    class_counts[class_id] += 1

                except ValueError:
                    invalid_lines += 1

        except Exception as error:
            print("Read error:", file, error)

    print("Image files:", len(image_files))
    print("Label directories:")
    for folder in label_dirs:
        print(" ", folder.relative_to(root))

    print("Label files:", len(label_files))
    print("Empty label files:", empty_files)
    print("Annotations by class ID:", dict(sorted(class_counts.items())))
    print("Invalid annotation lines:", invalid_lines)
