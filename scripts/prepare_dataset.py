"""
Prepare a unified candidate YOLO dataset from:
  datasets/construction_ppe
  datasets/mendeley_ppe

This script:
- Preserves the original datasets.
- Maps both datasets to one class list.
- Excludes the ambiguous Construction PPE class "none".
- Drops malformed/out-of-range/zero-area boxes.
- Ignores orphan label files (labels with no matching image).
- Keeps source train/val/test splits; reserves a deterministic 10% of
  Mendeley train images as test because Mendeley has no test split.
- Skips exact duplicate images across the combined dataset.
- Writes a CSV report so results can be reviewed before training.

Run from the project root:
  python scripts/prepare_dataset.py
"""

from __future__ import annotations

import csv
import hashlib
import random
import shutil
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parents[1]
CONSTRUCTION = ROOT / "datasets" / "construction_ppe"
MENDELEY = ROOT / "datasets" / "mendeley_ppe"
OUTPUT = ROOT / "datasets" / "processed_ppe"

# Final class IDs. Both source datasets are mapped into this shared order.
CLASS_NAMES = [
    "person",
    "helmet",
    "no_helmet",
    "vest",
    "no_vest",
    "boots",
    "no_boots",
    "gloves",
    "no_gloves",
    "goggles",
    "no_goggle",
]
NAME_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# Source class ID -> final class name
CONSTRUCTION_MAP = {
    0: "helmet",
    1: "gloves",
    2: "vest",
    3: "boots",
    4: "goggles",
    # 5 ("none") intentionally excluded: its meaning is unclear.
    6: "person",
    7: "no_helmet",
    8: "no_goggle",
    9: "no_gloves",
    10: "no_boots",
}
MENDELEY_MAP = {
    0: "helmet",
    1: "no_helmet",
    2: "no_vest",
    3: "vest",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
RANDOM_SEED = 42
MENDELEY_TRAIN_TEST_FRACTION = 0.10


def image_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_images(folder: Path) -> dict[str, Path]:
    if not folder.exists():
        return {}
    return {
        path.stem: path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def parse_and_map_label(
    label_path: Path | None,
    source_map: dict[int, str],
) -> tuple[list[str], int, int, int]:
    """Return mapped label lines and counts: malformed, invalid, excluded."""
    output_lines: list[str] = []
    malformed = invalid = excluded = 0

    if label_path is None or not label_path.exists():
        return output_lines, malformed, invalid, excluded

    for raw_line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            malformed += 1
            continue

        try:
            source_class = int(float(parts[0]))
            x, y, width, height = map(float, parts[1:])
        except ValueError:
            malformed += 1
            continue

        if source_class not in source_map:
            excluded += 1
            continue

        # YOLO normalized coordinates must be within [0, 1], with positive size.
        if (
            not all(0.0 <= value <= 1.0 for value in (x, y, width, height))
            or width <= 0.0
            or height <= 0.0
            or x - width / 2 < 0.0
            or y - height / 2 < 0.0
            or x + width / 2 > 1.0
            or y + height / 2 > 1.0
        ):
            invalid += 1
            continue

        final_name = source_map[source_class]
        final_id = NAME_TO_ID[final_name]
        output_lines.append(
            f"{final_id} {x:.6f} {y:.6f} {width:.6f} {height:.6f}"
        )

    return output_lines, malformed, invalid, excluded


def collect_source_items(
    source_name: str,
    source_root: Path,
    split_name: str,
    image_folder_name: str,
    label_folder_name: str,
    source_map: dict[int, str],
) -> list[dict]:
    image_dir = source_root / image_folder_name
    label_dir = source_root / label_folder_name
    images = find_images(image_dir)
    items = []

    for stem, image_path in images.items():
        label_path = label_dir / f"{stem}.txt"
        items.append(
            {
                "source": source_name,
                "source_split": split_name,
                "image": image_path,
                "label": label_path if label_path.exists() else None,
                "source_map": source_map,
            }
        )
    return items


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(
            f"\nOutput folder already exists:\n{OUTPUT}\n"
            "To avoid overwriting data, rename or remove that processed_ppe "
            "folder after checking it, then run this script again."
        )

    required = [
        CONSTRUCTION / "images" / "train",
        CONSTRUCTION / "labels" / "train",
        MENDELEY / "train" / "images",
        MENDELEY / "train" / "labels",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Required dataset folders not found:\n" + "\n".join(missing))

    # Keep existing Ultralytics splits.
    items: list[dict] = []
    for split in ("train", "val", "test"):
        items.extend(
            collect_source_items(
                "construction_ppe",
                CONSTRUCTION,
                split,
                f"images/{split}",
                f"labels/{split}",
                CONSTRUCTION_MAP,
            )
        )

    # Keep Mendeley's validation split as validation.
    items.extend(
        collect_source_items(
            "mendeley_ppe",
            MENDELEY,
            "val",
            "valid/images",
            "valid/labels",
            MENDELEY_MAP,
        )
    )

    # Deterministically reserve 10% of Mendeley training images for test.
    mendeley_train = collect_source_items(
        "mendeley_ppe",
        MENDELEY,
        "train",
        "train/images",
        "train/labels",
        MENDELEY_MAP,
    )
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(mendeley_train)
    test_count = max(1, round(len(mendeley_train) * MENDELEY_TRAIN_TEST_FRACTION))
    for index, item in enumerate(mendeley_train):
        item["source_split"] = "test" if index < test_count else "train"
        items.append(item)

    # Create output structure only after input validation.
    for split in ("train", "val", "test"):
        (OUTPUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    report_rows = []
    totals = {
        "images_copied": 0,
        "duplicate_images_skipped": 0,
        "malformed_lines": 0,
        "invalid_boxes_removed": 0,
        "excluded_objects": 0,
        "images_without_label_file": 0,
    }

    for item in items:
        image_path: Path = item["image"]
        split = item["source_split"]
        source = item["source"]

        digest = image_hash(image_path)
        if digest in seen_hashes:
            totals["duplicate_images_skipped"] += 1
            report_rows.append(
                {
                    "source": source,
                    "source_split": split,
                    "image": str(image_path),
                    "status": "skipped_exact_duplicate",
                    "malformed_lines": 0,
                    "invalid_boxes_removed": 0,
                    "excluded_objects": 0,
                    "kept_objects": 0,
                }
            )
            continue
        seen_hashes.add(digest)

        if item["label"] is None:
            totals["images_without_label_file"] += 1

        mapped_lines, malformed, invalid, excluded = parse_and_map_label(
            item["label"], item["source_map"]
        )

        # Prefix filenames to avoid collisions between datasets/splits.
        output_stem = f"{source}_{split}_{image_path.stem}"
        destination_image = OUTPUT / "images" / split / f"{output_stem}{image_path.suffix.lower()}"
        destination_label = OUTPUT / "labels" / split / f"{output_stem}.txt"

        shutil.copy2(image_path, destination_image)
        destination_label.write_text(
            ("\n".join(mapped_lines) + "\n") if mapped_lines else "",
            encoding="utf-8",
        )

        totals["images_copied"] += 1
        totals["malformed_lines"] += malformed
        totals["invalid_boxes_removed"] += invalid
        totals["excluded_objects"] += excluded

        report_rows.append(
            {
                "source": source,
                "source_split": split,
                "image": str(image_path),
                "status": "copied",
                "malformed_lines": malformed,
                "invalid_boxes_removed": invalid,
                "excluded_objects": excluded,
                "kept_objects": len(mapped_lines),
            }
        )

    # Generate Ultralytics data.yaml.
    yaml_lines = [
        f"path: {OUTPUT.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    yaml_lines.extend(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    (OUTPUT / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    # Save a row-by-row preparation report.
    report_path = OUTPUT / "preparation_report.csv"
    with report_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)

    print("\nDATASET PREPARATION COMPLETE")
    print(f"Output: {OUTPUT}")
    for key, value in totals.items():
        print(f"{key}: {value}")
    print(f"Class count: {len(CLASS_NAMES)}")
    print(f"YAML: {OUTPUT / 'data.yaml'}")
    print(f"Report: {report_path}")
    print("\nIMPORTANT: This is a candidate dataset, not yet approved for training.")
    print("Review questionable class annotations and the preparation report first.")


if __name__ == "__main__":
    main()
