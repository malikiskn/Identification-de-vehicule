import cv2

from yolo_pipeline import yolo_predictions

# Charger le modèle
net = cv2.dnn.readNetFromONNX('./runs/train/Model/weights/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Ouvrir la vidéo source
cap = cv2.VideoCapture('../Test_video/v.mp4')

# Obtenir les infos de la vidéo d'origine
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Définir le writer pour exporter la vidéo annotée
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./exports/output_annotated.mp4', fourcc, fps, (width, height))
while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin de la vidéo.")
        break

    # Appliquer la détection
    result = yolo_predictions(frame, net)

    # Écrire la frame annotée dans la nouvelle vidéo
    out.write(result)

# Nettoyage
cap.release()
out.release()
cv2.destroyAllWindows()
cv2.waitKey(1)