"""
Fire Detection — SVM Inference
Run this AFTER svm_train.py has been executed.

Usage:
    python svm_predict.py
    
Then edit the bottom section to choose your use case:
    - Single image
    - Batch folder
"""

import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import local_binary_pattern

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMG_SIZE   = (160, 160)
SVM_PATH   = "./saved_models/svm_fire.pkl"
SCALER_PATH= "./saved_models/scaler.pkl"
LABELS     = {0: "NON-FIRE 🌿", 1: "FIRE 🔥"}


# ═══════════════════════════════════════════════════════════════════
# 1 — LOAD MODEL
# ═══════════════════════════════════════════════════════════════════

def load_model():
    with open(SVM_PATH, "rb") as f:
        svm = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print("[LOADED] SVM model and scaler ready.")
    return svm, scaler


# ═══════════════════════════════════════════════════════════════════
# 2 — PREPROCESSING
# ═══════════════════════════════════════════════════════════════════

def extract_hog(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    hog  = cv2.HOGDescriptor(
        _winSize=(160, 160), _blockSize=(16, 16),
        _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9
    )
    return hog.compute(gray).flatten()


def extract_lbp(img_array, P=8, R=1):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    lbp  = local_binary_pattern(gray, P, R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), density=True)
    return hist


def preprocess(image_path, scaler):
    """Load image → extract HOG+LBP → scale → return feature vector."""
    img  = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr  = np.array(img)
    feat = np.concatenate([extract_hog(arr), extract_lbp(arr)]).reshape(1, -1)
    return scaler.transform(feat)


# ═══════════════════════════════════════════════════════════════════
# 3 — SINGLE IMAGE PREDICTION
# ═══════════════════════════════════════════════════════════════════

def predict_image(image_path, svm, scaler):
    """
    Predict a single image.
    Prints result and shows the image with label.
    """
    feat  = preprocess(image_path, scaler)
    label = svm.predict(feat)[0]
    prob  = svm.predict_proba(feat)[0][label]

    print(f"\n  Image      : {image_path}")
    print(f"  Prediction : {LABELS[label]}")
    print(f"  Confidence : {prob * 100:.1f}%")

    img   = Image.open(image_path).convert("RGB")
    color = "red" if label == 1 else "green"
    plt.figure(figsize=(5, 4))
    plt.imshow(img)
    plt.title(f"{LABELS[label]}  ({prob*100:.1f}%)", color=color, fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return label, prob


# ═══════════════════════════════════════════════════════════════════
# 4 — BATCH FOLDER PREDICTION
# ═══════════════════════════════════════════════════════════════════

def predict_folder(folder_path, svm, scaler, show_grid=True):
    """
    Predict all images in a folder.
    Prints a summary table and shows a visual grid.
    """
    img_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    if not img_files:
        print("[WARN] No images found in folder.")
        return

    print(f"\n[INFO] Predicting {len(img_files)} images in '{folder_path}'...")
    print(f"\n{'File':<30} {'Prediction':<15} {'Confidence'}")
    print("-" * 55)

    results, preds, thumbs = [], [], []
    for fname in img_files:
        fpath = os.path.join(folder_path, fname)
        try:
            feat  = preprocess(fpath, scaler)
            label = svm.predict(feat)[0]
            prob  = svm.predict_proba(feat)[0][label]
            results.append((fname, LABELS[label], prob))
            preds.append(label)
            thumbs.append(Image.open(fpath).convert("RGB").resize((120, 120)))
            print(f"{fname:<30} {LABELS[label]:<15} {prob*100:.1f}%")
        except Exception as e:
            print(f"{fname:<30} [ERROR] {e}")

    # Summary
    n_fire = sum(preds)
    print(f"\n  Total: {len(preds)} | 🔥 Fire: {n_fire} | 🌿 Safe: {len(preds)-n_fire}")

    # Grid
    if show_grid and thumbs:
        cols = min(5, len(thumbs))
        rows = (len(thumbs) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
        axes = np.array(axes).flatten()
        for i, (img, pred) in enumerate(zip(thumbs, preds)):
            axes[i].imshow(img)
            axes[i].set_title(
                "FIRE🔥" if pred == 1 else "SAFE🌿",
                color="red" if pred == 1 else "green", fontsize=9
            )
            axes[i].axis("off")
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        plt.suptitle("SVM — Batch Predictions", fontsize=13)
        plt.tight_layout()
        plt.show()

    return results


# ═══════════════════════════════════════════════════════════════════
# MAIN — Edit below to choose your use case
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("       SVM FIRE DETECTION — INFERENCE")
    print("=" * 50)

    svm, scaler = load_model()

    # ── USE CASE 1: Single image ──────────────────────────────────
    predict_image("test_image.jpg", svm, scaler)   # <-- change path

    # ── USE CASE 2: Batch folder ──────────────────────────────────
    # predict_folder("my_test_images/", svm, scaler)
