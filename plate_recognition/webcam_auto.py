import cv2
from yolo_pipeline import yolo_predictions
from database import save_plate

# Charger le modèle
net = cv2.dnn.readNetFromONNX('../runs/train/Model/weights/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Trouver webcam dispo
cap = None
for i in range(5):
    test_cap = cv2.VideoCapture(i)
    if test_cap.isOpened():
        cap = test_cap
        break

if cap is None:
    print("❌ Aucune caméra détectée")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result_img, texts = yolo_predictions(frame, net)

    for plate in texts:
        if plate and plate != 'no number':
            save_plate(plate, source='webcam', db_name="detections.db")

    cv2.imshow('Webcam - Live Detection', result_img)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)