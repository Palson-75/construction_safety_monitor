
from pathlib import Path
import torch
from ultralytics import YOLO


def main():
    ROOT = Path(__file__).resolve().parents[1]

    DATA_YAML = ROOT / "datasets" / "processed_ppe" / "data.yaml"
    OUTPUT_DIR = ROOT / "outputs"

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected!")

    print("=" * 50)
    print("FULL YOLO26n TRAINING")
    print("=" * 50)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Dataset:", DATA_YAML)
    print("Training images: 2778")
    print("Validation images: 600")
    print("Test images: 324 (reserved for final evaluation)")

    model = YOLO("yolo26n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        amp=True,
        cache=False,
        patience=20,
        fraction=1.0,
        save=True,
        save_period=10,
        plots=True,
        project=str(OUTPUT_DIR),
        name="full_training",
        exist_ok=True
    )

    print("\nTRAINING COMPLETED!")
    print("Best model:", OUTPUT_DIR / "full_training" / "weights" / "best.pt")
    print("Last model:", OUTPUT_DIR / "full_training" / "weights" / "last.pt")


if __name__ == "__main__":
    main()
