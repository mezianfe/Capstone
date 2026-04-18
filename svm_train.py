"""
Fire Detection — SVM Training
Run this file to train and save the SVM model.

Usage:
    python svm_train.py

Output:
    saved_models/svm_fire.pkl
    saved_models/scaler.pkl
    fire_features.csv
    results/  (confusion matrix, ROC, PR, learning curve plots)
"""

import os
import pickle
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from skimage.feature import local_binary_pattern

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, precision_score,
    recall_score, f1_score
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FIRE_DIR     = "./fire_dataset/fire_images"
NON_FIRE_DIR = "./fire_dataset/non_fire_images"
IMG_SIZE     = (160, 160)
RANDOM_STATE = 42
CSV_PATH     = "./fire_features.csv"
SAVE_DIR     = "./saved_models"
RESULTS_DIR  = "./results/svm"
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
    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════════
# 2 — FEATURE EXTRACTION  (HOG + LBP)
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


def extract_features(X):
    features = []
    for i, img in enumerate(X):
        feat = np.concatenate([extract_hog(img), extract_lbp(img)])
        features.append(feat)
        if (i + 1) % 100 == 0:
            print(f"  Extracted {i+1}/{len(X)} images...")
    return np.array(features)


def save_csv(features, labels, path=CSV_PATH):
    df = pd.DataFrame(features)
    df["label"] = labels
    df.to_csv(path, index=False)
    print(f"[SAVED] Features CSV → {path}")


# ═══════════════════════════════════════════════════════════════════
# 3 — TRAINING
# ═══════════════════════════════════════════════════════════════════

def train(X_feat, y):
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)
    print("[INFO] Training SVM with RBF kernel...")
    svm = SVC(kernel="rbf", C=10, gamma="scale",
              probability=True, random_state=RANDOM_STATE)
    svm.fit(X_scaled, y)
    print("[DONE] SVM training complete.")
    return svm, scaler


# ═══════════════════════════════════════════════════════════════════
# 4 — EVALUATION & PLOTS
# ═══════════════════════════════════════════════════════════════════

def evaluate(svm, scaler, X_feat, y_test):
    X_scaled = scaler.transform(X_feat)
    y_pred   = svm.predict(X_scaled)
    y_prob   = svm.predict_proba(X_scaled)[:, 1]

    print("\n" + "=" * 50)
    print("SVM — EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy  : {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"Precision : {precision_score(y_test, y_pred, average='weighted') * 100:.2f}%")
    print(f"Recall    : {recall_score(y_test, y_pred, average='weighted') * 100:.2f}%")
    print(f"F1-Score  : {f1_score(y_test, y_pred, average='weighted') * 100:.2f}%")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred, target_names=["non-fire", "fire"]))
    return y_pred, y_prob


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["non-fire", "fire"],
                yticklabels=["non-fire", "fire"])
    plt.title("SVM — Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.show()


def plot_roc(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc     = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"SVM — ROC Curve (AUC = {roc_auc:.2f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=150)
    plt.show()


def plot_precision_recall(y_test, y_prob):
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, color="steelblue")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("SVM — Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "precision_recall.png"), dpi=150)
    plt.show()


def plot_learning_curve(svm, X_feat, y):
    print("[INFO] Computing learning curve (this may take a moment)...")
    train_sizes, train_scores, val_scores = learning_curve(
        SVC(kernel="rbf", C=10, gamma="scale", probability=True),
        X_feat, y, cv=5, scoring="accuracy",
        train_sizes=np.linspace(0.625, 1.0, 8),
        random_state=RANDOM_STATE
    )
    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_scores.mean(axis=1), label="Training Score")
    plt.plot(train_sizes, val_scores.mean(axis=1),   label="Cross-Validation Score")
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy")
    plt.title("SVM — Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "learning_curve.png"), dpi=150)
    plt.show()


# ═══════════════════════════════════════════════════════════════════
# 5 — SAVE MODEL
# ═══════════════════════════════════════════════════════════════════

def save_model(svm, scaler):
    with open(os.path.join(SAVE_DIR, "svm_fire.pkl"), "wb") as f:
        pickle.dump(svm, f)
    with open(os.path.join(SAVE_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"[SAVED] SVM   → {SAVE_DIR}/svm_fire.pkl")
    print(f"[SAVED] Scaler→ {SAVE_DIR}/scaler.pkl")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("       SVM FIRE DETECTION — TRAINING")
    print("=" * 50)

    print("\n[1/5] Loading images...")
    X, y = load_images(FIRE_DIR, NON_FIRE_DIR)
    print(f"      Total: {len(X)} | Fire: {y.sum()} | Non-fire: {len(y)-y.sum()}")

    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"      Train: {len(X_train)} | Test: {len(X_test)}")

    print("\n[2/5] Extracting HOG + LBP features...")
    feat_train = extract_features(X_train)
    feat_test  = extract_features(X_test)
    save_csv(feat_train, y_train)

    print("\n[3/5] Training SVM...")
    svm, scaler = train(feat_train, y_train)

    print("\n[4/5] Evaluating...")
    y_pred, y_prob = evaluate(svm, scaler, feat_test, y_test)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc(y_test, y_prob)
    plot_precision_recall(y_test, y_prob)
    plot_learning_curve(svm, feat_train, y_train)

    print("\n[5/5] Saving model...")
    save_model(svm, scaler)

    print("\n✅ Done! All results saved to ./results/svm/")
