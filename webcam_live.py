import cv2
from yolo_pipeline import yolo_predictions

# Charger le modèle YOLOv5
net = cv2.dnn.readNetFromONNX('./runs/train/Model/weights/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Ouvrir la webcam (0 = caméra par défaut)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : impossible d'accéder à la webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur lors de la lecture de la webcam.")
        break

    # Appliquer la détection en direct
    result = yolo_predictions(frame, net)

    # Afficher le résultat dans une fenêtre
    cv2.imshow('Webcam - Détection YOLO', result)

    # Appuyer sur Échap pour quitter
    if cv2.waitKey(1) == 27:
        break

# Libérer la caméra et fermer la fenêtre
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)