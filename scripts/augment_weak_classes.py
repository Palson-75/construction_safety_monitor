
from pathlib import Path
import shutil
import cv2
import random
import yaml

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "datasets" / "processed_ppe"
DEST = ROOT / "datasets" / "processed_ppe_augmented"

IMAGE_DIR = SOURCE / "images" / "train"
LABEL_DIR = SOURCE / "labels" / "train"

TARGET_CLASSES = {6, 8, 10}

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]

random.seed(42)


def find_image(stem):
    for ext in IMAGE_EXTENSIONS:
        path = IMAGE_DIR / (stem + ext)
        if path.exists():
            return path
    return None


def augment_image(image, variant):
    if variant == 1:
        # Horizontal flip
        image = cv2.flip(image, 1)

        # Brightness and contrast
        image = cv2.convertScaleAbs(
            image, alpha=1.10, beta=8
        )

    elif variant == 2:
        # Different brightness and contrast
        image = cv2.convertScaleAbs(
            image, alpha=0.90, beta=15
        )

    return image


def transform_labels(lines, variant):
    output = []

    for line in lines:
        parts = line.split()

        if len(parts) != 5:
            continue

        cls, x, y, w, h = map(float, parts)

        if variant == 1:
            x = 1.0 - x

        output.append(
            f"{int(cls)} {x:.8f} {y:.8f} "
            f"{w:.8f} {h:.8f}"
        )

    return output


def main():

    if DEST.exists():
        raise FileExistsError(
            f"{DEST} already exists. "
            "Do not overwrite it automatically."
        )

    print("Copying original processed dataset...")

    shutil.copytree(SOURCE, DEST)

    matching = {}

    for label_file in LABEL_DIR.glob("*.txt"):

        lines = label_file.read_text().splitlines()

        present = {
            int(line.split()[0])
            for line in lines
            if len(line.split()) == 5
        }

        weak_classes = present.intersection(TARGET_CLASSES)

        if not weak_classes:
            continue

        # Two variants for no_boots images.
        # One variant for no_gloves/no_goggle images.
        variants = 2 if 6 in weak_classes else 1

        matching[label_file.stem] = variants

    print("Matching training images:", len(matching))

    generated = 0

    for stem, number_of_variants in matching.items():

        image_path = find_image(stem)
        label_path = LABEL_DIR / f"{stem}.txt"

        if image_path is None:
            print("Image missing:", stem)
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print("Could not read:", image_path)
            continue

        lines = label_path.read_text().splitlines()

        for variant in range(1, number_of_variants + 1):

            augmented = augment_image(image.copy(), variant)
            new_labels = transform_labels(lines, variant)

            new_stem = f"{stem}_aug{variant}"

            output_image = (
                DEST / "images" / "train" /
                f"{new_stem}{image_path.suffix}"
            )

            output_label = (
                DEST / "labels" / "train" /
                f"{new_stem}.txt"
            )

            if not cv2.imwrite(str(output_image), augmented):
                raise RuntimeError(
                    f"Failed to save {output_image}"
                )

            output_label.write_text(
                "\n".join(new_labels) + "\n"
            )

            generated += 1

    # Update the copied YAML to use the new dataset location.
    config = {
        "path": str(DEST).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 11,
        "names": {
            0: "person",
            1: "helmet",
            2: "no_helmet",
            3: "vest",
            4: "no_vest",
            5: "boots",
            6: "no_boots",
            7: "gloves",
            8: "no_gloves",
            9: "goggles",
            10: "no_goggle"
        }
    }

    with open(DEST / "data.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print("\nAUGMENTATION COMPLETED")
    print("Generated images:", generated)
    print("New dataset:", DEST)
    print("Original dataset remains unchanged.")


if __name__ == "__main__":
    main()
