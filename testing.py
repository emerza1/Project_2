import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.models import load_model

# Try a few common model file names to keep things flexible
CANDIDATE_MODELS = [
    "models/defect_cnn_v1.keras",
    "models/aircraft_defect_model.h5",
    "models/defect_cnn_v1.h5",
]

# Class names must match the training generator order
CLASS_NAMES = ["crack", "missing-head", "paint-off"]

# Images to test, paths can be adjusted as needed
TEST_FILES = {
    "crack": "Data/test/crack/test_crack.jpg",
    "missing-head": "Data/test/missing-head/test_missinghead.jpg",
    "paint-off": "Data/test/paint-off/test_paintoff.jpg",
}

IMG_H = 500
IMG_W = 500

def load_first_available(paths):
    """Try loading the first model file that exists."""
    for p in paths:
        if os.path.exists(p):
            try:
                print(f"Loading model from: {p}")
                mdl = load_model(p)
                print("Model loaded")
                return mdl
            except Exception as err:
                print(f"Could not load {p}: {err}")
    raise FileNotFoundError("No valid model file found in CANDIDATE_MODELS")

try:
    model = load_first_available(CANDIDATE_MODELS)
except Exception as e:
    print(f"Error: {e}")
    print("Train your model first, or update CANDIDATE_MODELS with the right path")
    raise SystemExit(1)

def prepare_image(path, target_size=(IMG_H, IMG_W)):
    """Read, resize, scale to [0,1], add batch dimension."""
    img = kimage.load_img(path, target_size=target_size)
    arr = kimage.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def softmax_to_label(probs, labels):
    """Pick the top class and confidence from probabilities."""
    idx = int(np.argmax(probs))
    return labels[idx], float(probs[idx])

print("\n" + "=" * 60)
print("RUNNING PREDICTIONS ON SAMPLE IMAGES")
print("=" * 60)

os.makedirs("outputs", exist_ok=True)

# Collect predictions for confusion matrix and F2
y_true = []
y_pred = []

for true_name, img_path in TEST_FILES.items():
    print(f"\nImage: {img_path}")

    if not os.path.exists(img_path):
        print(f"Missing file: {img_path}")
        continue

    batch = prepare_image(img_path, target_size=(IMG_H, IMG_W))
    probs = model.predict(batch, verbose=0)[0]
    pred_name, pred_conf = softmax_to_label(probs, CLASS_NAMES)

    # Record for metrics
    y_true.append(CLASS_NAMES.index(true_name))
    y_pred.append(CLASS_NAMES.index(pred_name))

    # Print a small summary to console
    print(f"True label: {true_name}")
    print(f"Predicted: {pred_name} ({pred_conf * 100:.1f}%)")
    print("Class probabilities:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {probs[i] * 100:.1f}%")

    # Make a figure with the photo and a small overlay
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(kimage.load_img(img_path))
    ax.axis("off")

    # Prepare overlay text
    overlay = [
        f"True: {true_name}",
        f"Pred: {pred_name}",
        "",
        f"{CLASS_NAMES[0]}: {probs[0] * 100:.1f}%",
        f"{CLASS_NAMES[1]}: {probs[1] * 100:.1f}%",
        f"{CLASS_NAMES[2]}: {probs[2] * 100:.1f}%",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(overlay),
        transform=ax.transAxes,
        fontsize=14,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        color="green" if pred_name == true_name else "red",
        weight="bold",
    )

    # Title and save
    good = pred_name == true_name
    plt.title("Correct prediction" if good else "Incorrect prediction",
              fontsize=16, fontweight="bold", color="green" if good else "red", pad=20)

    out_path = f"outputs/pred_{true_name}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out_path}")

# Build and print confusion matrix
print("\n" + "=" * 60)
print("CONFUSION MATRIX")
num_classes = len(CLASS_NAMES)
cm = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t, p] += 1

if cm.sum() == 0:
    print("No test images found, check TEST_FILES paths")
else:
    print("Rows are true labels, columns are predicted labels:")
    print(cm)

    # Compute F2 per class, macro and micro
    beta = 2.0
    beta2 = beta * beta
    f2_per_class = []

    print("\nF2 SCORES PER CLASS:")
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        denom = (beta2 * precision) + recall
        f2 = ((1 + beta2) * precision * recall / denom) if denom > 0 else 0.0
        f2_per_class.append(f2)

        print(f"  {CLASS_NAMES[c]}: precision={precision:.3f}; recall={recall:.3f}; F2={f2:.3f}")

    macro_f2 = float(np.mean(f2_per_class)) if len(f2_per_class) else 0.0

    # Micro F2 equals accuracy for single label multiclass
    total = cm.sum()
    correct = np.trace(cm)
    micro_precision = correct / total if total > 0 else 0.0
    micro_recall = micro_precision
    micro_f2 = micro_precision

    print(f"\nMacro F2: {macro_f2:.3f}")
    print(f"Micro F2: {micro_f2:.3f}")

    # Save a confusion matrix image
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(num_classes)); ax.set_yticklabels(CLASS_NAMES)
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_cm = "outputs/confusion_matrix.png"
    plt.savefig(out_cm, dpi=300)
    plt.close()
    print(f"Saved figure: {out_cm}")

# Legacy accuracy block kept for compatibility
hits = 0
total = 0
for true_name, img_path in TEST_FILES.items():
    if not os.path.exists(img_path):
        continue
    total += 1
    probs = model.predict(prepare_image(img_path), verbose=0)[0]
    pred_name, _ = softmax_to_label(probs, CLASS_NAMES)
    if pred_name == true_name:
        hits += 1

if total == 0:
    print("No test images found, check TEST_FILES paths")
else:
    acc = 100.0 * hits / total
    print(f"Accuracy on sample set: {hits}/{total} correct ({acc:.1f}%)")

print("\nResults saved in the outputs folder")

