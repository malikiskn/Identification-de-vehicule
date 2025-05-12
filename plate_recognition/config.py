# Fichier central pour tous les paramètres ajustables

# Dimensions d'entrée du modèle YOLO
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# Seuils pour la détection et NMS
CONFIDENCE_THRESHOLD = 0.4     # seuil minimum pour considérer une détection
CLASS_SCORE_THRESHOLD = 0.3    # seuil de probabilité pour la classe "license_plate"
NMS_THRESHOLD = 0.3           # suppression des doublons
# Chemin du modèle YOLOv5
MODEL_PATH = 'runs/train/Model/weights/best.onnx'