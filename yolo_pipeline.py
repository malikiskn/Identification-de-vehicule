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
def get_detections(img,net):
    # 1.CONVERT IMAGE TO YOLO FORMAT
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row,col)
    input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
    input_image[0:row,0:col] = image

    # 2. GET PREDICTION FROM YOLO MODEL
    blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WIDTH,INPUT_HEIGHT),swapRB=True,crop=False)
    net.setInput(blob)
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
def non_maximum_supression(input_image,detections):
    
    # 3. FILTER DETECTIONS BASED ON CONFIDENCE AND PROBABILIY SCORE
    
    # center x, center y, w , h, conf, proba
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/INPUT_WIDTH
    y_factor = image_h/INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4] # confidence of detecting license plate
        if confidence > 0.4:
            class_score = row[5] # probability score of license plate
            if class_score > 0.25:
                cx, cy , w, h = row[0:4]

                left = int((cx - 0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height])

                confidences.append(confidence)
                boxes.append(box)

    # 4.1 CLEAN
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    
    # 4.2 NMS
    index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45)
    
    return boxes_np, confidences_np, index

# Cette fonction dessine sur l’image les boîtes de détection des plaques,
# le score de confiance (en haut), et le texte lu par OCR (en bas).
# Elle utilise les résultats de YOLO (boxes, confiances) + Tesseract (OCR).
def drawings(image,boxes_np,confidences_np,index):
    # 5. Drawings
    for ind in index:
        x,y,w,h =  boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf*100)
        license_text = extract_text(image,boxes_np[ind])


        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.rectangle(image,(x,y-30),(x+w,y),(255,0,255),-1)
        cv2.rectangle(image,(x,y+h),(x+w,y+h+25),(0,0,0),-1)


        cv2.putText(image,conf_text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
        cv2.putText(image,license_text,(x,y+h+27),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),1)

    return image

# Fonction principale de prédiction. Elle applique les 3 étapes :
# - Détection avec YOLOv5
# - Filtrage avec suppression non maximale (NMS)
# - Dessin des résultats sur l’image (boîtes + OCR)
# Elle retourne l’image annotée.
def yolo_predictions(img,net):
    # step-1: detections
    input_image, detections = get_detections(img,net)
    # step-2: NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    # step-3: Drawings
    result_img = drawings(img,boxes_np,confidences_np,index)
    return result_img
# Cette fonction utilise Tesseract OCR pour lire le texte contenu dans une boîte (bbox).
# Elle extrait la région de l’image correspondant à la plaque,
# vérifie qu’elle est valide, puis retourne le texte lu (ou 'no number' si vide).
def extract_text(image,bbox):
    x,y,w,h = bbox
    roi = image[y:y+h, x:x+w]

    if 0 in roi.shape:
        return 'no number'

    else:
        text = pytesseract.image_to_string(roi)
        text = text.strip()

        return text