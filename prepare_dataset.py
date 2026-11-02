
from pathlib import Path
import random
import shutil

random.seed(42)

PROJECT = Path(__file__).resolve().parent
RAW = PROJECT / "data" / "raw"
OUTPUT = PROJECT / "data" / "images"

classes = {
    "real": RAW / "Real",
    "fake": RAW / "Fake",
}

extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

for label, source in classes.items():
    if not source.is_dir():
        raise FileNotFoundError(f"Folder not found: {source}")

    files = [
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]

    if not files:
        raise ValueError(f"No images found in {source}")

    random.shuffle(files)

    train_end = int(len(files) * 0.70)
    val_end = train_end + int(len(files) * 0.15)

    splits = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:],
    }

    for split, split_files in splits.items():
        destination = OUTPUT / split / label
        destination.mkdir(parents=True, exist_ok=True)

        for index, file in enumerate(split_files):
            target = destination / f"{label}_{index:06d}{file.suffix.lower()}"
            shutil.copy2(file, target)

        print(f"{split}/{label}: {len(split_files)} images")

print("\nDataset preparation completed.")
print(f"Output: {OUTPUT}")
