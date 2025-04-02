# Fichier central pour tous les paramètres ajustables

# Dimensions d'entrée du modèle YOLO
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# Seuils pour la détection et NMS
CONFIDENCE_THRESHOLD = 0.5     # seuil minimum pour considérer une détection
CLASS_SCORE_THRESHOLD = 0.2    # seuil de probabilité pour la classe "license_plate"
NMS_THRESHOLD = 0.4            # suppression des doublons