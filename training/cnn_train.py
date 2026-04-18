"""
Fire Detection — CNN (MobileNetV2) Training
Run this file to train and save the CNN model.

Usage:
    python cnn_train.py

Output:
    saved_models/mobilenetv2_fire.h5
    results/cnn/  (loss curve, accuracy curve, confusion matrix, ROC, PR curve)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    precision_score, recall_score, f1_score
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FIRE_DIR     = "./fire_dataset/fire_images"
NON_FIRE_DIR = "./fire_dataset/non_fire_images"
IMG_SIZE     = (160, 160)
BATCH_SIZE   = 32
EPOCHS       = 25
RANDOM_STATE = 42
SAVE_DIR     = "./saved_models"
RESULTS_DIR  = "./results/cnn"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# 1 — DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_images(fire_dir, non_fire_dir, img_size=IMG_SIZE):
    images, labels = [], []
    for path, label in [(fire_dir, 1), (non_fire_dir, 0)]:
        for fname in os.listdir(path):
            fpath = os.path.join(path, fname)
            try:
                img = Image.open(fpath).convert("RGB").resize(img_size)
                images.append(np.array(img))
                labels.append(label)
            except Exception:
                pass
    return np.array(images), np.array(labels)


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125,   # ~10% of total
        random_state=RANDOM_STATE, stratify=y_train
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ═══════════════════════════════════════════════════════════════════
# 2 — DATA GENERATORS  (augmentation)
# ═══════════════════════════════════════════════════════════════════

def get_generators(X_train, y_train, X_val, y_val):
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2]
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_flow = train_gen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    val_flow   = val_gen.flow(X_val,   y_val,   batch_size=BATCH_SIZE, shuffle=False)
    return train_flow, val_flow


# ═══════════════════════════════════════════════════════════════════
# 3 — MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════

def build_model():
    """
    MobileNetV2 (ImageNet pretrained, frozen)
    + GlobalAveragePooling → Dropout → Dense(sigmoid)
    """
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    base.trainable = False   # freeze pretrained layers

    x      = base.output
    x      = GlobalAveragePooling2D()(x)
    x      = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    return model


# ═══════════════════════════════════════════════════════════════════
# 4 — TRAINING
# ═══════════════════════════════════════════════════════════════════

def train(model, train_flow, val_flow):
    cb = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    print("[INFO] Training MobileNetV2...")
    history = model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=EPOCHS,
        callbacks=[cb],
        verbose=1
    )
    print("[DONE] Training complete.")
    return history


# ═══════════════════════════════════════════════════════════════════
# 5 — EVALUATION
# ═══════════════════════════════════════════════════════════════════

def evaluate(model, X_test, y_test):
    X_scaled = X_test / 255.0
    y_prob   = model.predict(X_scaled, verbose=0).flatten()
    y_pred   = (y_prob >= 0.5).astype(int)

    print("\n" + "=" * 50)
    print("CNN (MobileNetV2) — EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy  : {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"Precision : {precision_score(y_test, y_pred, average='weighted') * 100:.2f}%")
    print(f"Recall    : {recall_score(y_test, y_pred, average='weighted') * 100:.2f}%")
    print(f"F1-Score  : {f1_score(y_test, y_pred, average='weighted') * 100:.2f}%")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred, target_names=["non-fire", "fire"]))
    return y_pred, y_prob


# ═══════════════════════════════════════════════════════════════════
# 6 — PLOTS
# ═══════════════════════════════════════════════════════════════════

def plot_loss_accuracy(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"],     label="training_loss",    color="blue")
    axes[0].plot(history.history["val_loss"], label="val_loss",         color="orange")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"],     label="training_accuracy", color="blue")
    axes[1].plot(history.history["val_accuracy"], label="val_accuracy",      color="orange")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_accuracy.png"), dpi=150)
    plt.show()
    print("[SAVED] loss_accuracy.png")


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=["non-fire", "fire"],
                yticklabels=["non-fire", "fire"])
    plt.title("CNN — Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.show()
    print("[SAVED] confusion_matrix.png")


def plot_roc(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc     = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="darkorange")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"CNN — ROC Curve (AUC = {roc_auc:.2f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=150)
    plt.show()
    print("[SAVED] roc_curve.png")


def plot_precision_recall(y_test, y_prob):
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, color="darkorange")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("CNN — Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "precision_recall.png"), dpi=150)
    plt.show()
    print("[SAVED] precision_recall.png")


# ═══════════════════════════════════════════════════════════════════
# 7 — SAVE MODEL
# ═══════════════════════════════════════════════════════════════════

def save_model(model):
    path = os.path.join(SAVE_DIR, "mobilenetv2_fire.h5")
    model.save(path)
    print(f"[SAVED] CNN model → {path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("   CNN (MobileNetV2) FIRE DETECTION — TRAINING")
    print("=" * 50)

    print("\n[1/5] Loading images...")
    X, y = load_images(FIRE_DIR, NON_FIRE_DIR)
    print(f"      Total: {len(X)} | Fire: {y.sum()} | Non-fire: {len(y)-y.sum()}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"      Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print("\n[2/5] Preparing data generators...")
    train_flow, val_flow = get_generators(X_train, y_train, X_val, y_val)

    print("\n[3/5] Building MobileNetV2 model...")
    model = build_model()

    print("\n[4/5] Training...")
    history = train(model, train_flow, val_flow)

    print("\n[5/5] Evaluating and saving...")
    y_pred, y_prob = evaluate(model, X_test, y_test)
    plot_loss_accuracy(history)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc(y_test, y_prob)
    plot_precision_recall(y_test, y_prob)
    save_model(model)

    print("\n✅ Done! All results saved to ./results/cnn/")
