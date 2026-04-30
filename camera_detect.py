import cv2
import numpy as np
import joblib

# Load trained model
model = joblib.load("fire_svm_model.pkl")

IMG_SIZE = 64

# Start webcam
cap = cv2.VideoCapture(0)

print("[INFO] Starting camera... Press Q to exit")

# For smoothing predictions
preds = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for model
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    # Convert to HSV (same as training)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Flatten for SVM
    flat_img = hsv_img.flatten().reshape(1, -1)

    # -----------------------------
    #  FIRE COLOR FILTER
    # -----------------------------
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 150, 150])
    upper = np.array([35, 255, 255])

    mask = cv2.inRange(hsv_frame, lower, upper)
    fire_pixels = cv2.countNonZero(mask)

    # -----------------------------
    #  PREDICTION LOGIC
    # -----------------------------
    if fire_pixels > 3000:
        prediction = model.predict(flat_img)[0]
    else:
        prediction = 0  # force SAFE if no fire-like color

    # -----------------------------
    #  SMOOTHING (anti-flicker)
    # -----------------------------
    preds.append(prediction)
    if len(preds) > 10:
        preds.pop(0)

    final_pred = round(sum(preds) / len(preds))

    # -----------------------------
    #  DISPLAY
    # -----------------------------
    if final_pred == 1:
        label = "🔥 FIRE DETECTED"
        color = (0, 0, 255)
    else:
        label = "SAFE"
        color = (0, 255, 0)

    # Draw box
    cv2.rectangle(frame, (10, 10), (350, 80), color, 2)

    # Show text
    cv2.putText(frame, label, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Optional: show fire mask (debug)
    cv2.imshow("Fire Mask", mask)

    cv2.imshow("Fire Detection (SVM)", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
