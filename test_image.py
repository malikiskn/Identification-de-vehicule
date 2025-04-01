import cv2
from yolo_pipeline import yolo_predictions

# Charger le modèle YOLOv5 (format ONNX)
net = cv2.dnn.readNetFromONNX('./runs/train/Model/weights/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Charger une image
image = cv2.imread('../Test_image/img2.jpg')

# Appliquer la prédiction
result = yolo_predictions(image, net)

# Afficher le résultat
cv2.imshow('Resultat', result)
cv2.waitKey(0)
cv2.destroyAllWindows()