
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABEL_ROOT = ROOT / "datasets" / "processed_ppe" / "labels"

fixed = 0
unchanged = 0

for split in ("train", "val", "test"):
    label_dir = LABEL_ROOT / split

    for path in label_dir.glob("*.txt"):
        lines = path.read_text().splitlines()
        new_lines = []

        for line in lines:
            parts = line.split()

            if len(parts) != 5:
                new_lines.append(line)
                continue

            try:
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                new_lines.append(line)
                continue

            left = max(0.0, x - w / 2)
            top = max(0.0, y - h / 2)
            right = min(1.0, x + w / 2)
            bottom = min(1.0, y + h / 2)

            new_w = right - left
            new_h = bottom - top

            if new_w <= 0 or new_h <= 0:
                new_lines.append(line)
                unchanged += 1
                continue

            new_x = (left + right) / 2
            new_y = (top + bottom) / 2

            new_line = (
                f"{cls} {new_x:.8f} {new_y:.8f} "
                f"{new_w:.8f} {new_h:.8f}"
            )

            if new_line != line:
                fixed += 1

            new_lines.append(new_line)

        path.write_text("\n".join(new_lines) + "\n")

print("Boundary correction completed.")
print("Annotations corrected:", fixed)
print("Annotations skipped:", unchanged)
