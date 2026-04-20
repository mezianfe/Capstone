import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

DATASET_PATH = "dataset"
IMG_SIZE = 64

data = []
labels = []

categories = ["fire", "no_fire"]

print("[INFO] Loading images...")

for category in categories:
    path = os.path.join(DATASET_PATH, category)
    label = 1 if category == "fire" else 0

    for img_name in os.listdir(path):
        try:
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Convert to HSV (better for fire detection)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Flatten image
            img = img.flatten()

            data.append(img)
            labels.append(label)

        except:
            pass

data = np.array(data)
labels = np.array(labels)

print("[INFO] Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

print("[INFO] Training SVM...")

model = SVC(kernel='rbf')  # better than linear
model.fit(X_train, y_train)

print("[INFO] Testing...")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("[INFO] Saving model...")

joblib.dump(model, "fire_svm_model.pkl")

print("✅ Training Complete!")