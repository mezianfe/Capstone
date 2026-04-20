import cv2
import numpy as np
import joblib

IMG_SIZE = 64

# Load model
model = joblib.load("fire_svm_model.pkl")

# Change this to your test image
img = cv2.imread("test.png")

if img is None:
    print("❌ Image not found")
    exit()

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img = img.flatten().reshape(1, -1)

prediction = model.predict(img)

if prediction == 1:
    print("🔥 Fire Detected")
else:
    print("✅ No Fire")