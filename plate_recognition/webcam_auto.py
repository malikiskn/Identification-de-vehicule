import cv2
from yolo_pipeline import yolo_predictions

# Charger le modèle YOLOv5
net = cv2.dnn.readNetFromONNX('../runs/train/Model/weights/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# 🔍 Recherche automatique de la première caméra disponible
cap = None
for i in range(5):  # essaie les indices de 0 à 4
    test_cap = cv2.VideoCapture(i)
    if test_cap.isOpened():
        print(f"✅ Caméra détectée à l'index {i}")
        cap = test_cap
        break
    test_cap.release()

if cap is None:
    print("❌ Aucune webcam disponible.")
    exit()

# 🎥 Boucle principale : lecture et détection en direct
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur lors de la lecture de la webcam.")
        break

    result_img, texts = yolo_predictions(frame, net)


    # Affichage en direct
    cv2.imshow('Webcam - Détection', result_img)
    print("Plaques détectées (live) :", texts)

    # Quitter avec la touche Échap
    if cv2.waitKey(1) == 27:
        break

# Libération des ressources
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)