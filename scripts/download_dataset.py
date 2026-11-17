
import urllib.request
import zipfile
import ssl
import certifi
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "datasets"
ZIP_PATH = DATASET_DIR / "construction-ppe.zip"
EXTRACT_DIR = DATASET_DIR / "construction_ppe"

URL = (
    "https://github.com/ultralytics/assets/releases/"
    "download/v0.0.0/construction-ppe.zip"
)


def download_dataset():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.iterdir()):
        print("Dataset folder already contains files.")
        print("Inspect it before downloading again.")
        return

    print("Downloading Construction PPE dataset...")

    context = ssl.create_default_context(
        cafile=certifi.where()
    )

    request = urllib.request.Request(
        URL,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    try:
        with urllib.request.urlopen(
            request, context=context, timeout=60
        ) as response:
            with open(ZIP_PATH, "wb") as file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    file.write(chunk)

        print("Download completed.")

        print("Extracting dataset...")
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)

        print("Extraction completed!")

        ZIP_PATH.unlink(missing_ok=True)

    except Exception as error:
        print("Download or extraction failed:", error)
        print("The ZIP file, if downloaded, has been kept for inspection.")


if __name__ == "__main__":
    download_dataset()
