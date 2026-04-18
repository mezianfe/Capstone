"""
Fire Detection — CNN (MobileNetV2) Inference
Run this AFTER cnn_train.py has been executed.

Usage:
    python cnn_predict.py

Then edit the bottom section to choose your use case:
    - Single image
    - Batch folder
    - Real-time webcam
    - Video file
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMG_SIZE   = (160, 160)
MODEL_PATH = "./saved_models/mobilenetv2_fire.h5"
LABELS     = {0: "NON-FIRE 🌿", 1: "FIRE 🔥"}


# ═══════════════════════════════════════════════════════════════════
# 1 — LOAD MODEL
# ═══════════════════════════════════════════════════════════════════

def load_model():
    model = keras_load(MODEL_PATH)
    print("[LOADED] CNN model ready.")
    return model


# ═══════════════════════════════════════════════════════════════════
# 2 — PREPROCESSING
# ═══════════════════════════════════════════════════════════════════

def preprocess(image_path):
    """Load image, resize, normalize → ready for CNN input."""
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)   # shape: (1, 160, 160, 3)


# ═══════════════════════════════════════════════════════════════════
# 3 — SINGLE IMAGE PREDICTION
# ═══════════════════════════════════════════════════════════════════

def predict_image(image_path, model, threshold=0.5):
    """
    Predict a single image.
    Shows the image with label + confidence.
    Returns (label, confidence).
    """
    x     = preprocess(image_path)
    prob  = model.predict(x, verbose=0)[0][0]
    label = 1 if prob >= threshold else 0
    conf  = prob if label == 1 else 1 - prob

    print(f"\n  Image      : {image_path}")
    print(f"  Prediction : {LABELS[label]}")
    print(f"  Confidence : {conf * 100:.1f}%")

    img   = Image.open(image_path).convert("RGB")
    color = "red" if label == 1 else "green"
    plt.figure(figsize=(5, 4))
    plt.imshow(img)
    plt.title(f"{LABELS[label]}  ({conf*100:.1f}%)", color=color, fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return label, conf


# ═══════════════════════════════════════════════════════════════════
# 4 — BATCH FOLDER PREDICTION
# ═══════════════════════════════════════════════════════════════════

def predict_folder(folder_path, model, threshold=0.5, show_grid=True):
    """
    Predict all images in a folder.
    Prints a summary table and shows a visual grid.
    Returns list of (filename, label, confidence).
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
            x     = preprocess(fpath)
            prob  = model.predict(x, verbose=0)[0][0]
            label = 1 if prob >= threshold else 0
            conf  = prob if label == 1 else 1 - prob
            results.append((fname, LABELS[label], conf))
            preds.append(label)
            thumbs.append(Image.open(fpath).convert("RGB").resize((120, 120)))
            print(f"{fname:<30} {LABELS[label]:<15} {conf*100:.1f}%")
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
        plt.suptitle("CNN — Batch Predictions", fontsize=13)
        plt.tight_layout()
        plt.show()

    return results


# ═══════════════════════════════════════════════════════════════════
# 5 — REAL-TIME WEBCAM / VIDEO DETECTION
# ═══════════════════════════════════════════════════════════════════

def realtime_detection(model, source=0, threshold=0.5):
    """
    Real-time fire detection from webcam or video file.

    Args:
        model     : loaded CNN model
        source    : 0 = webcam  |  "video.mp4" = video file
        threshold : fire confidence threshold (default 0.5)

    Controls:
        Q → quit
        S → save screenshot
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Cannot open video source.")
        return

    print("[INFO] Real-time detection started. Press Q to quit, S to screenshot.")
    frame_count = 0
    label, conf = 0, 0.0   # initial default

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predict every 3 frames for speed
        if frame_count % 3 == 0:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, IMG_SIZE)
            x       = np.expand_dims(resized / 255.0, axis=0)
            prob    = model.predict(x, verbose=0)[0][0]
            label   = 1 if prob >= threshold else 0
            conf    = prob if label == 1 else 1 - prob

        # Draw overlay
        color = (0, 0, 255) if label == 1 else (0, 180, 0)
        text  = f"{'FIRE' if label==1 else 'SAFE'}  {conf*100:.1f}%"
        cv2.rectangle(frame, (0, 0), (280, 50), color, -1)
        cv2.putText(frame, text, (8, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Red alert border when fire detected
        if label == 1:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)

        cv2.imshow("CNN Fire Detection — Q to quit", frame)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            name = f"screenshot_{frame_count}.jpg"
            cv2.imwrite(name, frame)
            print(f"[SAVED] {name}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Detection stopped.")


# ═══════════════════════════════════════════════════════════════════
# MAIN — Edit below to choose your use case
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("   CNN (MobileNetV2) FIRE DETECTION — INFERENCE")
    print("=" * 50)

    model = load_model()

    # ── USE CASE 1: Single image ──────────────────────────────────
    predict_image("test_image.jpg", model)         # <-- change path

    # ── USE CASE 2: Batch folder ──────────────────────────────────
    # predict_folder("my_test_images/", model)

    # ── USE CASE 3: Live webcam ───────────────────────────────────
    # realtime_detection(model, source=0)

    # ── USE CASE 4: Video file ────────────────────────────────────
    # realtime_detection(model, source="fire_video.mp4")
