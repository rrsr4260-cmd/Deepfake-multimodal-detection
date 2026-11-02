
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

# Project paths
ROOT = Path(r"D:\DeepGuard-X")

TRAIN_DIR = ROOT / "data" / "images" / "Train"
TEST_DIR = ROOT / "data" / "images" / "Test"

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "image_model.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 5
SEED = 42

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load training images
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    class_names=["Fake", "Real"],
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=SEED
)

# Load test images separately
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    class_names=["Fake", "Real"],
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print("Class mapping:", train_ds.class_names)
# Fake = 0, Real = 1

# Improve input pipeline performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# Pretrained image feature extractor
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)
base_model.trainable = False

# Build classifier
inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

model.summary()

# Train only on the training folder
model.fit(
    train_ds,
    epochs=EPOCHS
)

# Evaluate using the separate test folder
print("\nTest results:")
model.evaluate(test_ds)

# Save model for DeepGuard X
model.save(MODEL_PATH)
print(f"\nModel saved at: {MODEL_PATH}")
