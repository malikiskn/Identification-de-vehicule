import cv2
import numpy as np
import pytesseract
import platform
import pytesseract
import re
import difflib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plate_utils import is_valid_plate


# Spécifie le chemin de Tesseract selon le système d'exploitation
if platform.system() == 'Darwin':  # macOS
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
elif platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:  # Linux
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'



INPUT_WIDTH =  640
INPUT_HEIGHT = 640


'''
cette fonction transforme l’image pour YOLO et récupère les prédictions brutes du modèle.
'''
def get_detections(img, net):
    # Conserver les couleurs originales (supprimer la conversion en gris)
    image = img.copy()
    
    # Adapter l'image au format carré
    row, col, _ = image.shape
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    # Préparer l'image pour le modèle
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()[0]  # Prendre le premier output

    return input_image, detections

'''
Ce que cette fonction
*** Filtrer les prédictions de YOLO pour ne garder que les plus confiantes.
*** Convertir les coordonnées des boîtes à l’échelle de l’image.
*** Utiliser la Suppression Non Maximale (NMS) pour éliminer les doublons
Sans NMS, YOLO donne souvent plusieurs boîtes très proches pour une même plaque.
NMS garde seulement la meilleure boîte, ce qui rend les résultats propres.
'''

from config import INPUT_WIDTH, INPUT_HEIGHT, CONFIDENCE_THRESHOLD, CLASS_SCORE_THRESHOLD, NMS_THRESHOLD

def non_maximum_supression(input_image, detections):
    #Étape 3 : Filtrer les détections avec les bons seuils
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / INPUT_WIDTH
    y_factor = image_h / INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]

        if confidence > CONFIDENCE_THRESHOLD:
            class_score = row[5]
            if class_score > CLASS_SCORE_THRESHOLD:
                cx, cy, w, h = row[0:4]

                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])

                boxes.append(box)
                confidences.append(confidence)

    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()

    #Étape 4 : NMS avec seuil configurable
    index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, CLASS_SCORE_THRESHOLD, NMS_THRESHOLD)

    return boxes_np, confidences_np, index


# Cette fonction dessine sur l’image les boîtes de détection des plaques,
# le score de confiance (en haut), et le texte lu par OCR (en bas).
# Elle utilise les résultats de YOLO (boxes, confiances) + Tesseract (OCR).
def drawings(image, boxes_np, confidences_np, index):
    for ind in index:
        raw_text, cleaned_text = extract_text(image, boxes_np[ind], pad=2)
        license_text = cleaned_text if cleaned_text else 'Aucune lecture OCR'
        x, y, w, h = boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf * 100)

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 0, 255), -1)
        cv2.putText(image, conf_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.rectangle(image, (x, y + h), (x + w, y + h + 25), (0, 0, 0), -1)
        cv2.putText(image, license_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

    return image



# Fonction principale de prédiction. Elle applique les 3 étapes :
# - Détection avec YOLOv5
# - Filtrage avec suppression non maximale (NMS)
# - Dessin des résultats sur l’image (boîtes + OCR)
# Elle retourne l’image annotée et le texte lu.
# Elle utilise les fonctions get_detections, non_maximum_supression et drawings.


def yolo_predictions(img, net):
    input_image, detections = get_detections(img, net)
    boxes, confidences, indexes = non_maximum_supression(input_image, detections)

    result_img = img.copy()
    detected_texts = []

    for ind in indexes:
        raw_text, cleaned_text = extract_text(result_img, boxes[ind], pad=2)
        
        if is_valid_plate(cleaned_text):
            # Vérifier si la plaque est similaire à une déjà détectée
            is_new = True
            for existing in detected_texts:
                if difflib.SequenceMatcher(None, cleaned_text, existing).ratio() > 0.85:
                    is_new = False
                    break
            if is_new:
                detected_texts.append(cleaned_text)

    if not detected_texts:
        detected_texts.append('Aucune lecture OCR')

    result_img = drawings(result_img, boxes, confidences, indexes)
    return result_img, detected_texts

# Cette fonction utilise Tesseract OCR pour lire le texte contenu dans une boîte (bbox).
# Elle extrait la région de l’image correspondant à la plaque,
# vérifie qu’elle est valide, puis retourne le texte lu (ou 'no number' si vide).
def extract_text(image, bbox=None, pad=2):
    if bbox is None:
        x, y, w, h = 0, 0, image.shape[1], image.shape[0]
    else:
        x, y, w, h = bbox
    
    x = max(0, x + pad)
    y = max(0, y + pad)
    w = max(0, w - 2 * pad)
    h = max(0, h - 2 * pad)

    roi = image[y:y+h, x:x+w]
    if 0 in roi.shape:
        return '', ''

    # Essai 1: Méthode originale
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
    raw_text = pytesseract.image_to_string(roi, config=config).strip().upper()
    cleaned_text = re.sub(r'^[-\d]+|[-]+$', '', raw_text)

    # Essai 2: Seulement si échec du premier essai
    if not cleaned_text or len(cleaned_text) < 4:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        raw_text = pytesseract.image_to_string(binary, config=config).strip().upper()
        cleaned_text = re.sub(r'^[-\d]+|[-]+$', '', raw_text)

    # Post-traitement supplémentaire
    if cleaned_text:
        # Supprime les petits groupes de caractères isolés
        cleaned_text = ' '.join([word for word in cleaned_text.split() if len(word) > 2])
        
    return raw_text, cleaned_text if cleaned_text else 'Aucune lecture OCR'