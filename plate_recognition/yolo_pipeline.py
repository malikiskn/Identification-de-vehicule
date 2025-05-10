import cv2
import numpy as np
import pytesseract
import platform
import pytesseract

# Spécifie le chemin de Tesseract selon le système d'exploitation
if platform.system() == 'Darwin':  # macOS
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
elif platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:  # Linux
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'



INPUT_WIDTH =  640
INPUT_HEIGHT = 640
from skimage import io
import plotly.express as px

'''
cette fonction transforme l’image pour YOLO et récupère les prédictions brutes du modèle.
'''
def get_detections(img, net):
    #1. Prétraitement de l'image (augmentation du contraste)
    image = img.copy()
    # Convertit en niveaux de gris
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    # Améliore le contraste                       
    image = cv2.equalizeHist(image)
    # Revenir en BGR pour compatibilité modèle                                       
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)                        

    # 2. Adapter l'image au format carré YOLO
    row, col, d = image.shape
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    # 3. Préparer l'image pour le modèle
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)

    # 4. Obtenir les prédictions
    preds = net.forward()
    detections = preds[0]

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
        x, y, w, h = boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf * 100)

        # Lire le texte OCR depuis la plaque
        license_text = extract_text(image, boxes_np[ind])

        # Sécurité : si rien lu, afficher "NO TEXT"
        if license_text == '':
            license_text = 'NO TEXT'

        # Boîte principale (rose)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)

        # Fond rose pour la confiance
        cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 0, 255), -1)
        cv2.putText(image, conf_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # Fond noir pour le texte OCR
        cv2.rectangle(image, (x, y + h), (x + w, y + h + 25), (0, 0, 0), -1)
        cv2.putText(image, license_text, (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

    return image

# Fonction principale de prédiction. Elle applique les 3 étapes :
# - Détection avec YOLOv5
# - Filtrage avec suppression non maximale (NMS)
# - Dessin des résultats sur l’image (boîtes + OCR)
# Elle retourne l’image annotée et le texte lu.
# Elle utilise les fonctions get_detections, non_maximum_supression et drawings.


def yolo_predictions(img, net):
    input_image, detections = get_detections(img, net)
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)

    result_img = img.copy()
    texts = []

    for ind in index:
        text = extract_text(result_img, boxes_np[ind])
        if text.strip():  
            texts.append(text)

    result_img = drawings(result_img, boxes_np, confidences_np, index)

    return result_img, texts  


# Cette fonction utilise Tesseract OCR pour lire le texte contenu dans une boîte (bbox).
# Elle extrait la région de l’image correspondant à la plaque,
# vérifie qu’elle est valide, puis retourne le texte lu (ou 'no number' si vide).
def extract_text(image,bbox):
    x,y,w,h = bbox
    roi = image[y:y+h, x:x+w]

    if 0 in roi.shape:
        return 'no number'
    

    else :
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(roi, config=custom_config)
        text = text.strip()

        return text