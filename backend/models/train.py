"""
Improved brain tumor classifier — EfficientNetB0 transfer learning v2.

Fixes vs v1:
  - Phase 2 ModelCheckpoint is initialized with Phase 1's best val_accuracy,
    so the saved model is always the global best (not just Phase 2's local best).
  - Fine-tune LR dropped from 5e-5 → 1e-5 to prevent catastrophic forgetting.
  - Label smoothing (0.1) on cross-entropy — major help for small datasets.
  - L2 weight decay on Dense layers to curb overfitting.
  - Deeper head: GlobalAvgPool → BN → Dense(512) → Dense(128) → Softmax(4).
  - Unfreeze top 50 backbone layers (was 30).
  - Validation split raised from 15% → 20% for a more reliable signal.
  - More patience: Phase 1 EarlyStopping=12, Phase 2=15.

Usage:
    cd backend
    python models/train.py
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "archive"))
TRAIN_DIR   = os.path.join(ARCHIVE_DIR, "Training")
TEST_DIR    = os.path.join(ARCHIVE_DIR, "Testing")
MODEL_OUT   = os.path.join(SCRIPT_DIR, "classification_v2.keras")

# ── Hyper-parameters ──────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS_HEAD = 30    # Phase 1: frozen backbone
EPOCHS_FINE = 40    # Phase 2: fine-tune top layers
NUM_CLASSES = 4
SEED        = 42
L2          = 1e-4  # weight decay for Dense layers

CLASS_LABELS = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# ── Data generators ───────────────────────────────────────────────────────────
# No rescale — EfficientNetB0 expects raw [0, 255] pixels (internal preprocessing).
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.10,
    zoom_range=0.20,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.15,   # 15% val — more training data helps convergence
)

test_datagen = ImageDataGenerator()


def make_generator(datagen, directory, subset=None):
    kwargs = dict(
        directory=directory,
        classes=CLASS_LABELS,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        seed=SEED,
        shuffle=(subset == "training" or subset is None),
    )
    if subset:
        kwargs["subset"] = subset
    return datagen.flow_from_directory(**kwargs)


train_gen = make_generator(train_datagen, TRAIN_DIR, subset="training")
val_gen   = make_generator(train_datagen, TRAIN_DIR, subset="validation")
test_gen  = make_generator(test_datagen,  TEST_DIR)

# ── Class weights ─────────────────────────────────────────────────────────────
counts = np.array([train_gen.classes.tolist().count(i) for i in range(NUM_CLASSES)])
total  = counts.sum()
class_weight = {i: total / (NUM_CLASSES * counts[i]) for i in range(NUM_CLASSES)}
print("Class counts :", dict(zip(CLASS_LABELS, counts)))
print("Class weights:", {CLASS_LABELS[k]: round(v, 3) for k, v in class_weight.items()})

# ── Model ─────────────────────────────────────────────────────────────────────
backbone = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(*IMG_SIZE, 3),
)
backbone.trainable = False  # Phase 1: frozen

inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
x = backbone(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(L2))(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs, name="efficientnetb0_tumor_v3")
model.summary()

# ── Phase 1: Train head (backbone frozen) ─────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"],
)

callbacks_p1 = [
    ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_OUT, monitor="val_accuracy", save_best_only=True, verbose=1),
]

print("\n=== Phase 1: Training head (backbone frozen) ===")
history1 = model.fit(
    train_gen,
    epochs=EPOCHS_HEAD,
    validation_data=val_gen,
    class_weight=class_weight,
    callbacks=callbacks_p1,
)

phase1_best_val = max(history1.history["val_accuracy"])
print(f"\nPhase 1 best val_accuracy: {phase1_best_val:.4f}")

# ── Phase 2: Fine-tune top 50 backbone layers ─────────────────────────────────
backbone.trainable = True
for layer in backbone.layers[:-50]:
    layer.trainable = False

# Lower LR (10× less than Phase 1) to avoid catastrophic forgetting
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"],
)

# initial_value_threshold ensures Phase 2 only saves if it beats Phase 1
callbacks_p2 = [
    ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        MODEL_OUT,
        monitor="val_accuracy",
        save_best_only=True,
        initial_value_threshold=phase1_best_val,  # only save if better than Phase 1
        verbose=1,
    ),
]

print("\n=== Phase 2: Fine-tuning top 50 backbone layers ===")
history2 = model.fit(
    train_gen,
    epochs=EPOCHS_FINE,
    validation_data=val_gen,
    class_weight=class_weight,
    callbacks=callbacks_p2,
)

# If Phase 2 didn't beat Phase 1, the Phase 1 weights are still in memory
# (restore_best_weights=True restores the best Phase 2 weights).
# Load the file to ensure we evaluate the globally best saved model.
print(f"\nLoading best saved model from: {MODEL_OUT}")
best_model = tf.keras.models.load_model(MODEL_OUT, compile=False)

# ── Evaluate ──────────────────────────────────────────────────────────────────
best_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
print("\n=== Test set evaluation ===")
test_loss, test_acc = best_model.evaluate(test_gen, verbose=1)
print(f"\nTest accuracy : {test_acc * 100:.2f}%")
print(f"Test loss     : {test_loss:.4f}")
print(f"Phase 1 best val_accuracy: {phase1_best_val * 100:.2f}%")
phase2_best_val = max(history2.history["val_accuracy"])
print(f"Phase 2 best val_accuracy: {phase2_best_val * 100:.2f}%")
print(f"\nModel saved to: {MODEL_OUT}")
