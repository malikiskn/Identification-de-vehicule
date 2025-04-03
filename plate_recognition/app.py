from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
from yolo_pipeline import yolo_predictions
from database import save_plate
from datetime import datetime

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
RESULT_PATH = os.path.join('static', 'result.jpg')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 📍 Page d'accueil
@app.route('/')
def index():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "Aucun fichier envoyé"

    file = request.files['file']
    if file.filename == '':
        return "Nom de fichier vide"

    if not allowed_file(file.filename):
        return "❌ Format non supporté. Seuls JPG, JPEG et PNG sont autorisés."

    # 🔒 Le fichier est valide, on continue
    filename = file.filename
    upload_dir = app.config['UPLOAD_FOLDER']
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, filename)
    file.save(file_path)

    # 🔍 Détection
    image = cv2.imread(file_path)
    if image is None:
        return "❌ Fichier non lisible (peut-être .HEIC ou corrompu)"

    # 🔥 Suite détection YOLO (inchangée)
    from config import INPUT_WIDTH, INPUT_HEIGHT
    net = cv2.dnn.readNetFromONNX('../runs/train/Model/weights/best.onnx')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    result_img, texts = yolo_predictions(image, net)

    for plate in texts:
        if plate and plate != "no number":
            save_plate(plate, source='web-upload')

    cv2.imwrite(RESULT_PATH, result_img)

    return redirect(url_for('result', plates=",".join(texts)))

# 📸 Afficher le résultat
@app.route('/result')
def result():
    plates = request.args.get("plates", "")
    return render_template("result.html", plates=plates.split(','), result_path=RESULT_PATH)

if __name__ == '__main__':
    app.run(debug=True)