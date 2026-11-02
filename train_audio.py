
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import librosa
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ==================================================
# 1. PATHS AND SETTINGS
# ==================================================

ROOT = Path(r"D:\DeepGuard-X")
DATA_DIR = ROOT / "data" / "audio"

TRAIN_CSV = DATA_DIR / "train.csv"
DEV_CSV = DATA_DIR / "dev.csv"

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "audio_model.joblib"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
MAX_SECONDS = 10
N_MFCC = 20
RANDOM_STATE = 42

# Dataset label mapping:
# 0 = real, 1 = fake

# ==================================================
# 2. RESOLVE AUDIO PATHS
# ==================================================

def resolve_audio_path(csv_path):
    """
    Resolve paths stored in the CSV.

    Example:
    train/fake/model_01/train/audio/example.flac

    Expected location:
    D:/DeepGuard-X/data/audio/train/fake/model_01/
        train/audio/example.flac
    """

    relative_path = str(csv_path).strip().replace("\\", "/")
    relative_path = relative_path.lstrip("/")

    # First try the exact relative path from the CSV.
    candidate = DATA_DIR.joinpath(*relative_path.split("/"))

    if candidate.is_file():
        return candidate

    # Fallback: search for the exact filename.
    filename = Path(relative_path).name
    matches = list(DATA_DIR.rglob(filename))

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        # Prefer a match whose relative path ends with
        # the path specified in the CSV.
        suffix = relative_path.lower()

        for match in matches:
            actual = match.relative_to(DATA_DIR).as_posix().lower()

            if actual.endswith(suffix):
                return match

    return None


# ==================================================
# 3. LOAD CSV METADATA
# ==================================================

def load_metadata(csv_file):
    if not csv_file.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_file}"
        )

    df = pd.read_csv(csv_file)
    df.columns = [str(col).strip().lower() for col in df.columns]

    required_columns = {"path", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{csv_file.name} must contain columns "
            f"'path' and 'label'. Found: {list(df.columns)}"
        )

    df = df.dropna(subset=["path", "label"]).copy()

    # Validate numeric labels. Do not silently reverse them.
    numeric_labels = pd.to_numeric(df["label"], errors="coerce")

    if numeric_labels.isna().any():
        raise ValueError(
            f"{csv_file.name} contains non-numeric labels."
        )

    if not numeric_labels.isin([0, 1]).all():
        raise ValueError(
            f"{csv_file.name} contains labels other than 0 and 1."
        )

    df["label"] = numeric_labels.astype(int)

    print(f"\nLoading {csv_file.name}")
    print("Rows in CSV:", len(df))
    print("CSV label counts:")
    print(df["label"].value_counts().sort_index())

    records = []
    missing = 0

    for row in df.itertuples(index=False):
        audio_path = resolve_audio_path(row.path)

        if audio_path is None:
            missing += 1
            continue

        records.append({
            "path": audio_path,
            "label": int(row.label)
        })

    result = pd.DataFrame(records)

    print("Audio files matched:", len(result))
    print("Audio files not found:", missing)

    if result.empty:
        raise ValueError(
            f"No audio files matched {csv_file.name}. "
            "Check the folder structure and CSV paths."
        )

    if result["label"].nunique() != 2:
        raise ValueError(
            f"{csv_file.name} must contain both classes."
        )

    return result


# ==================================================
# 4. EXTRACT MFCC FEATURES
# ==================================================

def extract_features(audio_path):
    try:
        audio, _ = librosa.load(
            str(audio_path),
            sr=SAMPLE_RATE,
            mono=True,
            duration=MAX_SECONDS
        )

        if audio.size == 0:
            raise ValueError("Empty audio file")

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            n_fft=512,
            hop_length=160
        )

        # 20 means + 20 standard deviations = 40 features
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1)
        ])

        if not np.isfinite(features).all():
            raise ValueError("Invalid feature values")

        return features.astype(np.float32)

    except Exception as error:
        print(f"Could not process {audio_path}: {error}")
        return None


def build_features(metadata, dataset_name):
    X = []
    y = []

    total = len(metadata)

    for index, row in enumerate(
        metadata.itertuples(index=False), start=1
    ):
        features = extract_features(row.path)

        if features is not None:
            X.append(features)
            y.append(row.label)

        if index % 100 == 0 or index == total:
            print(
                f"{dataset_name}: processed "
                f"{index}/{total} files"
            )

    if not X:
        raise ValueError(
            f"No usable audio files in {dataset_name}."
        )

    return np.asarray(X), np.asarray(y, dtype=np.int32)


# ==================================================
# 5. LOAD TRAIN AND DEV DATA
# ==================================================

print("DeepGuard X - Audio Model Training")

train_metadata = load_metadata(TRAIN_CSV)
dev_metadata = load_metadata(DEV_CSV)

print("\nExtracting training audio features...")
X_train, y_train = build_features(
    train_metadata, "TRAIN"
)

print("\nExtracting development audio features...")
X_dev, y_dev = build_features(
    dev_metadata, "DEV"
)

print("\nTraining feature shape:", X_train.shape)
print("Development feature shape:", X_dev.shape)

print("\nTraining class counts:")
print("Real (0):", int(np.sum(y_train == 0)))
print("Fake (1):", int(np.sum(y_train == 1)))

# ==================================================
# 6. TRAIN RANDOM FOREST
# ==================================================

print("\nTraining Random Forest classifier...")

model = RandomForestClassifier(
    n_estimators=100,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training completed.")

# ==================================================
# 7. EVALUATE ON DEV DATA
# ==================================================

print("\nEvaluating on development data...")

predictions = model.predict(X_dev)

accuracy = accuracy_score(y_dev, predictions)

print(f"\nDevelopment accuracy: {accuracy:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_dev,
        predictions,
        labels=[0, 1],
        target_names=["Real", "Fake"],
        zero_division=0
    )
)

print("Confusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(
    y_dev,
    predictions,
    labels=[0, 1]
))

# ==================================================
# 8. SAVE MODEL AND METADATA
# ==================================================

model_package = {
    "model": model,
    "sample_rate": SAMPLE_RATE,
    "max_seconds": MAX_SECONDS,
    "n_mfcc": N_MFCC,
    "feature_count": 2 * N_MFCC,
    "label_mapping": {
        0: "real",
        1: "fake"
    }
}

joblib.dump(model_package, MODEL_PATH)

print("\n===================================")
print("Audio model saved successfully!")
print(f"Location: {MODEL_PATH}")
print("Labels: 0 = Real, 1 = Fake")
print("===================================")
