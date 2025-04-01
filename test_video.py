import cv2
from yolo_pipeline import yolo_predictions

# Charger le modèle
net = cv2.dnn.readNetFromONNX('./runs/train/Model/weights/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Lecture vidéo
cap = cv2.VideoCapture('../Test_video/v.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = yolo_predictions(frame, net)
    cv2.imshow('YOLO', results)
    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)