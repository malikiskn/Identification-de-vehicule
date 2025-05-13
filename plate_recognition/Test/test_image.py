import cv2
from yolo_pipeline import yolo_predictions
from database import save_plate

# Charger le modèle
net = cv2.dnn.readNetFromONNX('../runs/train/Model/weights/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Charger une image
image = cv2.imread('../../Test_image/img2.jpg')

# Prédiction
result_img, texts = yolo_predictions(image, net)

# Enregistrement dans la base
for plate in texts:
    if plate and plate != 'no number':
        save_plate(plate, source="image_test",db_name="detections.db")
        print("Plaques détectées :", texts)
from database import get_all_plates
for row in get_all_plates():
    print("📋", row)
# Affichage
cv2.imshow('Résultat', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)

'''
from database import get_all_plates

for row in get_all_plates():
    print(row)
'''
