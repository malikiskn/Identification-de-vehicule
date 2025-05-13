import cv2
from yolo_pipeline import yolo_predictions
from database import save_plate

# Charger le modèle
net = cv2.dnn.readNetFromONNX('../runs/train/Model/weights/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

cap = cv2.VideoCapture('../../Test_video/v.mp4')

video_name = 'vidéo'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result_img, texts = yolo_predictions(frame, net)

    for plate in texts:
        if plate and plate != 'no number':
            save_plate(plate, source=video_name, db_name="detections.db")

    cv2.imshow('YOLO Video', result_img)
    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)