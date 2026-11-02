
from pathlib import Path
import cv2
import numpy as np
import joblib
import tensorflow as tf

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(r"D:\DeepGuard-X")
VIDEO_DIR = ROOT / "data" / "videos"
IMAGE_MODEL_PATH = ROOT / "models" / "image_model.keras"
OUTPUT_PATH = ROOT / "models" / "video_model.joblib"

IMAGE_SIZE = (224, 224)
FRAMES_PER_VIDEO = 12

VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm"
}

# Label mapping: real = 0, fake = 1
DATASET_FOLDERS = {
    "YouTube-real": 0,
    "Celeb-real": 0,
    "Celeb-YouTube-fake": 1
}

# =====================================================
# EXTRACT FEATURES FROM VIDEO
# =====================================================

def extract_video_features(video_path, image_model):
    """
    Sample frames from a video and summarize image-model scores.

    The image model must return one sigmoid score per frame.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[WARNING] Cannot open: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        print(f"[WARNING] Empty or unreadable video: {video_path}")
        return None

    frame_positions = np.linspace(
        0,
        total_frames - 1,
        num=min(FRAMES_PER_VIDEO, total_frames),
        dtype=int
    )

    frames = []

    for position in frame_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, IMAGE_SIZE)
        frames.append(frame)

    cap.release()

    if not frames:
        print(f"[WARNING] No frames extracted: {video_path}")
        return None

    batch = np.asarray(frames, dtype=np.float32)

    # IMPORTANT:
    # Match this preprocessing to your image-model training.
    # If training used rescale=1./255, divide by 255 here.
    # If using EfficientNetB0's built-in preprocessing,
    # check your training pipeline before changing this.
    # batch = batch / 255.0

    scores = image_model.predict(batch, verbose=0).reshape(-1)

    # Summarize frame scores into video-level features.
    features = np.array([
        np.mean(scores),
        np.std(scores),
        np.min(scores),
        np.max(scores),
        np.median(scores),
        np.percentile(scores, 25),
        np.percentile(scores, 75),
        len(scores) / FRAMES_PER_VIDEO
    ], dtype=np.float32)

    return features


# =====================================================
# MAIN TRAINING
# =====================================================

def main():

    if not IMAGE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Image model not found: {IMAGE_MODEL_PATH}"
        )

    if not VIDEO_DIR.exists():
        raise FileNotFoundError(
            f"Video dataset folder not found: {VIDEO_DIR}"
        )

    print("[INFO] Loading image model...")
    image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)

    print("[INFO] Image model input:", image_model.input_shape)
    print("[INFO] Image model output:", image_model.output_shape)

    X = []
    y = []
    video_groups = []

    # -------------------------------------------------
    # Read labeled folders
    # -------------------------------------------------

    for folder_name, label in DATASET_FOLDERS.items():

        folder = VIDEO_DIR / folder_name

        if not folder.exists():
            print(f"[WARNING] Folder not found: {folder}")
            continue

        videos = sorted([
            path for path in folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
        ])

        class_name = "REAL" if label == 0 else "FAKE"

        print(
            f"\n[INFO] {folder_name}: "
            f"{len(videos)} videos ({class_name})"
        )

        for index, video_path in enumerate(videos, start=1):

            try:
                features = extract_video_features(
                    video_path,
                    image_model
                )

                if features is not None:
                    X.append(features)
                    y.append(label)

                    # Keep track of source folder for split control.
                    video_groups.append(folder_name)

                if index % 10 == 0:
                    print(
                        f"Processed {index}/{len(videos)} videos"
                    )

            except Exception as error:
                print(
                    f"[WARNING] Skipped {video_path.name}: {error}"
                )

    # -------------------------------------------------
    # Validate dataset
    # -------------------------------------------------

    if len(X) == 0:
        raise ValueError(
            "No videos were processed. Check your video folders "
            "and supported video formats."
        )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    video_groups = np.asarray(video_groups)

    print("\n[INFO] Total processed videos:", len(X))
    print("[INFO] Real videos:", int(np.sum(y == 0)))
    print("[INFO] Fake videos:", int(np.sum(y == 1)))
    print("[INFO] Feature matrix shape:", X.shape)

    if len(np.unique(y)) != 2:
        raise ValueError(
            "Both real and fake videos are required for training."
        )

    # -------------------------------------------------
    # Split by source folder where possible
    # -------------------------------------------------
    # This avoids putting videos from the same source folder
    # in both training and testing. With only two source groups,
    # however, it cannot provide a balanced independent test.
    unique_groups = np.unique(video_groups)

    if len(unique_groups) >= 3:
        test_group = unique_groups[-1]
        test_mask = video_groups == test_group
        train_mask = ~test_mask

        if (
            len(np.unique(y[train_mask])) == 2
            and len(np.unique(y[test_mask])) == 2
        ):
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            print(f"\n[INFO] Held-out source folder: {test_group}")
        else:
            print(
                "[WARNING] Source-folder split lacks both classes. "
                "Using a stratified video split instead."
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
    else:
        print(
            "\n[WARNING] Too few source folders for a robust "
            "source-independent test; using a stratified video split."
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

    # -------------------------------------------------
    # Train video classifier
    # -------------------------------------------------

    print("\n[INFO] Training video classifier...")

    video_model = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    video_model.fit(X_train, y_train)

    # -------------------------------------------------
    # Evaluate
    # -------------------------------------------------

    predictions = video_model.predict(X_test)

    print("\n========== VIDEO MODEL RESULTS ==========")

    print(
        "Accuracy:",
        round(accuracy_score(y_test, predictions), 4)
    )

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Real", "Fake"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ))

    # -------------------------------------------------
    # Save model package
    # -------------------------------------------------

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_package = {
        "model": video_model,
        "feature_names": [
            "mean_frame_score",
            "std_frame_score",
            "min_frame_score",
            "max_frame_score",
            "median_frame_score",
            "percentile_25",
            "percentile_75",
            "sampled_frame_ratio"
        ],
        "label_mapping": {
            0: "real",
            1: "fake"
        },
        "frames_per_video": FRAMES_PER_VIDEO,
        "image_size": IMAGE_SIZE
    }

    joblib.dump(model_package, OUTPUT_PATH)

    print("\n[SUCCESS] Video model saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
