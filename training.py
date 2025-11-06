import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Image and training settings
IMG_H = 500
IMG_W = 500
N_CH = 3
INPUT_SHAPE = (IMG_H, IMG_W, N_CH)
BATCH = 32
N_CLASSES = 3   # crack, missing-head, paint-off
EPOCHS_MAX = 30

print(f"Input shape: {INPUT_SHAPE}")
print(f"Batch size: {BATCH}")
print(f"Classes: {N_CLASSES}")

# Folders
TRAIN_PATH = "Data/train"
VAL_PATH = "Data/valid"
TEST_PATH = "Data/test"

print("\nChecking folders")
if not os.path.exists(TRAIN_PATH):
    print(f"ERROR: missing {TRAIN_PATH}")
    exit(1)
else:
    print(f"Found: {TRAIN_PATH}")

if not os.path.exists(VAL_PATH):
    print(f"ERROR: missing {VAL_PATH}")
    exit(1)
else:
    print(f"Found: {VAL_PATH}")

if not os.path.exists(TEST_PATH):
    print(f"ERROR: missing {TEST_PATH}")
    exit(1)
else:
    print(f"Found: {TEST_PATH}")

# Data augmentation for train, simple rescale for val
print("\nBuilding data generators")

train_aug = ImageDataGenerator(
    rescale=1.0 / 255.0,
    shear_range=0.25,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    brightness_range=[0.85, 1.15],
    fill_mode="nearest",
)

val_aug = ImageDataGenerator(rescale=1.0 / 255.0)

try:
    train_gen = train_aug.flow_from_directory(
        TRAIN_PATH,
        target_size=(IMG_H, IMG_W),
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=True,
        seed=123,
    )
    print(f"Train images: {train_gen.n}")
except Exception as e:
    print(f"Train generator error: {e}")
    exit(1)

try:
    val_gen = val_aug.flow_from_directory(
        VAL_PATH,
        target_size=(IMG_H, IMG_W),
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=False,
        seed=123,
    )
    print(f"Validation images: {val_gen.n}")
except Exception as e:
    print(f"Validation generator error: {e}")
    exit(1)

print("\n" + "=" * 60)
print("DATA SUMMARY")
print("=" * 60)
print(f"Train samples: {train_gen.n}")
print(f"Val samples: {val_gen.n}")
print(f"Classes map: {train_gen.class_indices}")
print(f"Steps per epoch train: {train_gen.n // BATCH}")
print(f"Steps per epoch val: {val_gen.n // BATCH}")
print("=" * 60)

# Sanity checking the train count
expected_train = 1942
expected_val = 431
if train_gen.n == expected_train:
    print(f"Train count matches: {expected_train}")
else:
    print(f"Train count is {train_gen.n}, expected {expected_train}")

if val_gen.n == expected_val:
    print(f"Val count matches: {expected_val}")
else:
    print(f"Val count is {val_gen.n}, expected {expected_val}")

cnn = Sequential(name="defect_cnn")

print("Building CNN")

# Block 1
cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE, name="b1_conv"))
cnn.add(MaxPooling2D((2, 2), name="b1_pool"))

# Block 2
cnn.add(Conv2D(64, (3, 3), activation="relu", name="b2_conv"))
cnn.add(MaxPooling2D((2, 2), name="b2_pool"))

# Block 3
cnn.add(Conv2D(128, (3, 3), activation="relu", name="b3_conv"))
cnn.add(MaxPooling2D((2, 2), name="b3_pool"))

# Block 4
cnn.add(Conv2D(128, (3, 3), activation="relu", name="b4_conv"))
cnn.add(MaxPooling2D((2, 2), name="b4_pool"))

# Block 5
cnn.add(Conv2D(256, (3, 3), activation="relu", name="b5_conv"))
cnn.add(MaxPooling2D((2, 2), name="b5_pool"))

# Screw head
cnn.add(Flatten(name="flatten"))
cnn.add(Dense(256, activation="relu", name="head_dense1"))
cnn.add(Dropout(0.5, name="head_drop1"))
cnn.add(Dense(128, activation="relu", name="head_dense2"))
cnn.add(Dropout(0.5, name="head_drop2"))
cnn.add(Dense(N_CLASSES, activation="softmax", name="logits"))

print("\nModel overview")
cnn.summary()

print("Loss: categorical_crossentropy")
print("Optimizer: Adam with lr 0.0005")
print("Metrics: accuracy")

opt = Adam(learning_rate=0.0005)

cnn.compile(
    loss="categorical_crossentropy",
    optimizer=opt,
    metrics=["accuracy"],
)

print("\nHyperparameters")
print("Conv blocks: 32, 64, 128, 128, 256")
print("Kernel: 3x3, pooling 2x2")
print("Dense: 256 then 128, dropout 0.5")
print(f"Batch size: {BATCH}")
print(f"Max epochs: {EPOCHS_MAX}")
print(f"Image size: {IMG_H} x {IMG_W} x {N_CH}")

print("Setting callbacks")
hooks = [
    EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1),
]
print("Early stopping patience: 8")
print("Reduce LR on plateau factor: 0.5, patience: 4")

print("\nTraining plan")
print(f"Epochs: {EPOCHS_MAX}")
print(f"Train steps: {train_gen.n // BATCH}")
print(f"Val steps: {val_gen.n // BATCH}")

print("\nStarting training")
start_time = time.time()
print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

history = cnn.fit(
    train_gen,
    epochs=EPOCHS_MAX,
    validation_data=val_gen,
    callbacks=hooks,
    verbose=1,
)

end_time = time.time()
elapsed = end_time - start_time
hh = int(elapsed // 3600)
mm = int((elapsed % 3600) // 60)
ss = int(elapsed % 60)

print("\nTraining done")
print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total time: {hh}h {mm}m {ss}s, {elapsed:.2f} seconds")
print(f"Average per epoch: {elapsed / max(1, len(history.history.get('loss', []))):.2f} seconds")

# Saving the model
os.makedirs("models", exist_ok=True)
model_path = "models/defect_cnn_v1.keras"
cnn.save(model_path)
print(f"Saved model to: {model_path}")

# Plotting results
print("\nGenerating plots")

os.makedirs("outputs", exist_ok=True)

plt.figure(figsize=(14, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train", linewidth=2)
plt.plot(history.history["val_accuracy"], label="val", linewidth=2)
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train", linewidth=2)
plt.plot(history.history["val_loss"], label="val", linewidth=2)
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = "outputs/train_curves.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved plot to: {fig_path}")

# Final numbers
print("\n" + "=" * 60)
print("FINAL METRICS")
print("=" * 60)
tr_acc = history.history["accuracy"][-1]
va_acc = history.history["val_accuracy"][-1]
tr_loss = history.history["loss"][-1]
va_loss = history.history["val_loss"][-1]

print(f"Train acc: {tr_acc:.4f} or {tr_acc * 100:.2f}%")
print(f"Val acc: {va_acc:.4f} or {va_acc * 100:.2f}%")
print(f"Train loss: {tr_loss:.4f}")
print(f"Val loss: {va_loss:.4f}")

if tr_acc - va_acc > 0.15:
    print("\nHeads up, this may be overfitting, try more dropout, heavier aug, or a smaller net")
elif va_acc >= tr_acc:
    print("\nNice, generalization looks good")
else:
    print("\nLearning signal looks fine")

print("\nPipeline complete")
