import cv2
import numpy as np
import joblib

# Load trained model
model = joblib.load("fire_svm_model.pkl")

IMG_SIZE = 64

# Open webcam
cap = cv2.VideoCapture(0)

print("[INFO] Starting camera... Press Q to exit")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    # Convert to HSV (same as training!)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Flatten
    img = img.flatten().reshape(1, -1)

    # Prediction
    prediction = model.predict(img)

    # Label
    if prediction == 1:
        label = "🔥 FIRE DETECTED"
        color = (0, 0, 255)
    else:
        label = "SAFE"
        color = (0, 255, 0)

    # Show label on screen
    cv2.putText(frame, label, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Fire Detection (SVM)", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()