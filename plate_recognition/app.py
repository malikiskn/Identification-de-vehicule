from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import cv2
from yolo_pipeline import yolo_predictions
from database import save_plate
from datetime import datetime
from database import get_connection
import io
from flask import send_file
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import csv
from flask import Response
import time  
from yolo_pipeline import extract_text
from werkzeug.utils import secure_filename
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plate_utils import is_valid_plate
import difflib



app = Flask(__name__)

app.secret_key = "supersecretkey"
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
STATIC_FOLDER = os.path.join(os.path.dirname(__file__), 'static')
RESULT_IMG_PATH = os.path.join(STATIC_FOLDER, 'result.jpg')
RAW_VIDEO_PATH = os.path.join(STATIC_FOLDER, 'raw_video.avi')  # temporaire
RESULT_VIDEO_PATH = os.path.join(STATIC_FOLDER, 'result_video.mp4')  # final
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'MP4'}
EXPORT_FOLDER = os.path.join(STATIC_FOLDER, 'exports')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPORT_FOLDER, exist_ok=True)



app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_model():
    net = cv2.dnn.readNetFromONNX(os.path.join(os.path.dirname(__file__), '../runs/train/Model/weights/best.onnx'))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def get_available_camera_index(max_index=5):
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            cap.release()
            return index
        cap.release()
    return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Accueil
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    file = request.files['file']
    if not allowed_file(file.filename):
        return "❌ Format non supporté."

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    image = cv2.imread(file_path)
    net = load_model()
    result_img, texts = yolo_predictions(image, net)

    # Filtrage final pour l'affichage
    valid_plates = [plate for plate in texts if is_valid_plate(plate)]
    
    if not valid_plates:
        valid_plates = ['Aucune plaque valide détectée']

    from datetime import datetime
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    cleaned_plate = None
    for plate in texts:
        if plate and plate != 'no number':
            cleaned_plate = plate.replace(" ", "").replace("-", "").upper()
            break

    if cleaned_plate:
        image_filename = f"{cleaned_plate}_{now_str}.jpg"
    else:
        image_filename = f"image_{now_str}.jpg"

    image_path = os.path.join(EXPORT_FOLDER, image_filename)
    cv2.imwrite(image_path, result_img)

    if os.path.exists(file_path):
        os.remove(file_path)

    for plate in texts:
        if plate and plate != 'no number':
            save_plate(plate, source='image', image_path=f"exports/{image_filename}")

    # Sauvegarde pour affichage dans result.html
    cv2.imwrite(RESULT_IMG_PATH, result_img)

    
    import shutil
    shutil.copy(RESULT_IMG_PATH, os.path.join(EXPORT_FOLDER, 'result.jpg'))

    return redirect(url_for('result', media_type='image', plates=",".join(texts), video_name=image_filename))

@app.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files['file']
    if not allowed_file(file.filename):
        return "❌ Vidéo non supportée."

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    
    net = load_model()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(RAW_VIDEO_PATH, fourcc, fps, (width, height))

    plates_dict = {}  # Dictionnaire pour stocker les plaques uniques
    frame_count = 0
    last_valid_plate = None
    plate_counter = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        result_img, new_texts = yolo_predictions(frame, net)
        out.write(result_img)

        # Filtrer et compter les plaques valides
        for plate in new_texts:
            if is_valid_plate(plate):
                cleaned_plate = clean_plate_text(plate)
                
                # Vérifier la similarité avec les plaques déjà détectées
                is_new_plate = True
                for existing_plate in plates_dict.keys():
                    # Utiliser un seuil de similarité de 85%
                    if difflib.SequenceMatcher(None, cleaned_plate, existing_plate).ratio() > 0.85:
                        is_new_plate = False
                        cleaned_plate = existing_plate  # Garder la version existante
                        break
                
                if is_new_plate:
                    plates_dict[cleaned_plate] = plate  # Garder la version originale
                    plate_counter[cleaned_plate] = 1
                else:
                    plate_counter[cleaned_plate] += 1

    cap.release()
    out.release()

    # Ne garder que les plaques détectées plusieurs fois (pour éviter les faux positifs)
    final_plates = []
    for plate, count in plate_counter.items():
        if count >= 3:  # Seuil minimum de détections pour considérer la plaque comme valide
            final_plates.append(plates_dict[plate])

    if not final_plates:
        final_plates = ['Aucune plaque valide détectée']

    # Génération du nom de fichier
    main_plate = clean_plate_text(final_plates[0]) if final_plates else None
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_name = f"{main_plate}_{now_str}.mp4" if main_plate else f"video_{now_str}.mp4"
    video_path = os.path.join('static', 'exports', video_name)

    # Conversion vidéo
    os.system(f"ffmpeg -y -i {RAW_VIDEO_PATH} -vcodec libx264 -crf 23 {video_path}")
    if os.path.exists(RAW_VIDEO_PATH):
        os.remove(RAW_VIDEO_PATH)

    # Sauvegarde en base
    for plate in final_plates:
        if plate not in ['Aucune plaque valide détectée']:
            save_plate(plate, source='video', image_path=f"exports/{video_name}")

    # Nettoyage
    if os.path.exists(input_path):
        os.remove(input_path)

    return redirect(url_for('result', 
                         media_type='video',
                         plates=",".join(final_plates),
                         video_name=video_name))

def clean_plate_text(text):
    """Nettoie le texte de la plaque pour comparaison"""
    if not text:
        return ""
    return ''.join(c for c in str(text).upper() if c.isalnum() or c == '-')

@app.route('/use_webcam', methods=['POST'])
def use_webcam():
    from datetime import datetime

    cap = cv2.VideoCapture(0)
    net = load_model()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = 15

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(RAW_VIDEO_PATH, fourcc, fps, (width, height))

    plates_dict = {}  # Dictionnaire pour stocker les plaques uniques
    plate_counter = {}  # Compteur d'occurrences par plaque
    frame_count = 0
    min_duration = fps * 5      # Minimum 5 secondes
    max_duration = fps * 15     # Maximum 15 secondes
    no_plate_counter = 0
    no_plate_limit = fps * 2    # Si 2 sec sans plaque → stop

    while frame_count < max_duration:
        ret, frame = cap.read()
        if not ret:
            break

        result_img, new_texts = yolo_predictions(frame, net)
        out.write(result_img)
        frame_count += 1

        # Traitement des plaques détectées
        has_valid_plate = False
        for plate in new_texts:
            if is_valid_plate(plate):
                has_valid_plate = True
                cleaned = clean_plate_text(plate)
                
                # Vérifier la similarité avec les plaques existantes
                is_new = True
                for existing in plates_dict.keys():
                    if difflib.SequenceMatcher(None, cleaned, existing).ratio() > 0.85:
                        is_new = False
                        cleaned = existing  # Garder la version existante
                        break
                
                if is_new:
                    plates_dict[cleaned] = plate  # Garde la version originale
                    plate_counter[cleaned] = 1
                else:
                    plate_counter[cleaned] += 1

        if not has_valid_plate:
            no_plate_counter += 1
        else:
            no_plate_counter = 0

        if frame_count > min_duration and no_plate_counter > no_plate_limit:
            #print("🛑 Arrêt anticipé : plus de plaques détectées.")
            break

    cap.release()
    out.release()

    # Filtrer les plaques détectées au moins 3 fois
    final_plates = []
    for plate, count in plate_counter.items():
        if count >= 3:  # Seuil minimum de détections
            final_plates.append(plates_dict[plate])

    if not final_plates:
        final_plates = ['Aucune plaque valide détectée']

    # Génération du nom de fichier
    main_plate = clean_plate_text(final_plates[0]) if final_plates else None
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    if main_plate:
        final_name = f"{main_plate}_{now_str}.mp4"
    else:
        final_name = f"webcam_{now_str}.mp4"

    result_path = os.path.join('static', 'exports', final_name)

    # Conversion vidéo
    ffmpeg_cmd = f"ffmpeg -y -i {RAW_VIDEO_PATH} -vcodec libx264 -crf 23 {result_path}"
    os.system(ffmpeg_cmd)

    if os.path.exists(RAW_VIDEO_PATH):
        os.remove(RAW_VIDEO_PATH)

    # Sauvegarde en base (uniquement les plaques valides et uniques)
    for plate in final_plates:
        if plate and plate.upper() not in ['AUCUNE PLAQUE VALIDE DETECTEE']:
            save_plate(plate, source='webcam', image_path=f"exports/{final_name}")

    return redirect(url_for('result', 
                         media_type='video',
                         plates=",".join(final_plates),
                         video_name=final_name))
# Résultat
import time

@app.route('/result')
def result():
    plates = request.args.get("plates", "").split(",")
    media_type = request.args.get("media_type", "image")
    video_name = request.args.get("video_name", "result_video.mp4")

    return render_template(
        "result.html",
        media_type=media_type,
        plates=plates,
        video_name=video_name,
        time=time.time  
    )



# Variable globale pour stocker les plaques détectées en direct
live_plates = []

@app.route('/stream')
def stream():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    # Trouver la caméra disponible
    camera_index = get_available_camera_index()
    if camera_index is None:
        print("❌ Aucune caméra disponible.")
        return

    cap = cv2.VideoCapture(camera_index)
    net = load_model()
    global live_plates
    live_plates = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        # YOLO + annotation
        result_img, texts = yolo_predictions(frame, net)

        for plate in texts:
            if plate and plate != 'no number':
                if plate not in live_plates:
                    live_plates.append(plate)
                    save_plate(plate, source='webcam')

        # Encoder en JPEG
        ret, buffer = cv2.imencode('.jpg', result_img)
        frame = buffer.tobytes()

        # Yield frame pour MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
        
from flask import jsonify

@app.route('/get_live_plates')
def get_live_plates():
    global live_plates
    clean = [p for p in live_plates if p.upper() not in ['NO TEXT', 'NO NUMBER']]
    return jsonify({'plates': clean})

@app.route('/webcam_live')
def webcam_live():
    return render_template('webcam_live.html')

from database import get_all_plates


from collections import Counter
from database import get_all_plates

@app.route('/history')
def history():
    from database import get_connection

    source = request.args.get('source')
    conn = get_connection()
    cur = conn.cursor()

    if source:
        cur.execute("SELECT * FROM plates WHERE source = ? ORDER BY timestamp DESC", (source,))
    else:
        cur.execute("SELECT * FROM plates ORDER BY timestamp DESC")

    plates = cur.fetchall()
    conn.close()

    # Compter par source pour les graphiques
    from collections import Counter
    source_counts = Counter([p[2] for p in plates])
    from datetime import datetime
    date_counts = Counter([p[3][:10] for p in plates])
    count_total = len(plates)
    count_image = len([p for p in plates if p[2] == 'image'])
    count_video = len([p for p in plates if p[2] == 'video'])
    count_webcam = len([p for p in plates if p[2] == 'webcam'])
    return render_template('history.html',
                       plates=plates,
                       source_counts=source_counts,
                       date_counts=date_counts,
                       selected_source=source,
                       count_total=count_total,
                       count_image=count_image,
                       count_video=count_video,
                       count_webcam=count_webcam)
    
@app.route('/export-pdf')
def export_pdf():
    from database import get_all_plates
    rows = get_all_plates()

    # Création PDF en mémoire
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Titre
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, height - 50, "📋 Historique des plaques détectées")

    # En-têtes de colonne
    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, height - 80, "ID")
    p.drawString(100, height - 80, "Plaque")
    p.drawString(250, height - 80, "Source")
    p.drawString(400, height - 80, "Date")

    # Contenu
    p.setFont("Helvetica", 10)
    y = height - 100
    for row in rows:
        p.drawString(50, y, str(row[0]))
        p.drawString(100, y, row[1])
        p.drawString(250, y, row[2])
        p.drawString(400, y, row[3])
        y -= 20
        if y < 50:
            p.showPage()
            y = height - 50

    p.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="historique_plaques.pdf", mimetype='application/pdf')


import csv

@app.route('/export-csv')
def export_csv():
    from database import get_all_plates
    rows = get_all_plates()

    # Créer le contenu CSV en mémoire
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Plaque', 'Source', 'Date'])

    for row in rows:
        writer.writerow(row)

    output.seek(0)

    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment;filename=plaques_detectees.csv'}
    )


@app.route('/admin')
def admin():
    if not session.get('admin'):
        return redirect(url_for('login'))

    from database import get_all_plates
    rows = get_all_plates()

    source = request.args.get("source")
    if source:
        rows = [row for row in rows if row[2] == source]

    # Compteurs
    count_total = len(rows)
    count_image = sum(1 for r in rows if r[2] == "image")
    count_video = sum(1 for r in rows if r[2] == "video")
    count_webcam = sum(1 for r in rows if r[2] == "webcam")

    # Données graphiques
    from collections import Counter
    source_counts = Counter([r[2] for r in rows])
    date_counts = Counter([r[3][:10] for r in rows])

    return render_template('admin.html',
        plates=rows,
        selected_source=source,
        count_total=count_total,
        count_image=count_image,
        count_video=count_video,
        count_webcam=count_webcam,
        source_counts=source_counts,
        date_counts=date_counts
    )

@app.route('/delete_plate/<int:id>', methods=['POST'])
def delete_plate(id):
    print(f"Tentative de suppression de la plaque ID {id}")
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM plates WHERE id = ?", (id,))
    conn.commit()
    conn.close()
    flash("Plaque supprimée avec succès.")
    return redirect(url_for('admin'))

@app.route('/delete_all', methods=['POST'])
def delete_all():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM plates")
    conn.commit()
    conn.close()
    flash("Toutes les plaques ont été supprimées.")
    return redirect(url_for('admin'))

ADMIN_PASSWORD = "admin"  

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form.get('password')
        if password == ADMIN_PASSWORD:
            session['admin'] = True
            return redirect(url_for('admin'))
        else:
            flash("❌ Mot de passe incorrect")
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('admin', None)
    return redirect(url_for('index'))


# Affiche le formulaire de modification
@app.route('/edit/<int:id>')
def edit_plate(id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM plates WHERE id = ?", (id,))
    plate = cur.fetchone()
    conn.close()

    return render_template("edit_plate.html", plate=plate)

# Enregistre la plaque modifiée
@app.route('/update/<int:id>', methods=['POST'])
def update_plate_route(id):
    from_page = request.args.get("from_page", "history")
    new_plate = request.form.get("new_plate", "").strip()

    if not new_plate:
        flash("🚫 Le champ de la plaque est vide.", "danger")
        return redirect(url_for(from_page))

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE plates SET plate = ? WHERE id = ?", (new_plate, id))
    conn.commit()
    conn.close()

    flash(f"Plaque modifiée : {new_plate}", "success")
    return redirect(url_for(from_page))

@app.route('/add_plate', methods=['GET', 'POST'])
def add_plate():
    if request.method == 'POST':
        plate = request.form['plate']
        source = request.form['source']

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO plates (plate, source, timestamp) VALUES (?, ?, ?)", (plate, source, timestamp))
        conn.commit()
        conn.close()

        flash("Nouvelle plaque ajoutée avec succès.")
        return redirect(url_for('admin'))

    return render_template('add_plate.html')


@app.route('/detail/<plate>')
def vehicle_detail(plate):
    from database import get_connection
    from vehicle_info import get_vehicle_details  
    import os

    # Connexion à la base
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM plates WHERE plate=? ORDER BY timestamp DESC", (plate,))
    rows = cur.fetchall()
    conn.close()

    # Trouver une image liée
    image_path = None
    export_dir = os.path.join(app.static_folder, 'exports')
    for file in os.listdir(export_dir):
        if plate.replace(" ", "").replace("-", "").lower() in file.lower():
            image_path = 'exports/' + file
            break
    print("Image trouvée pour la plaque :", image_path)
    # Récupérer les infos de l'API simulée
    vehicle_info = get_vehicle_details(plate)

    return render_template("vehicle_info.html", plate=plate, vehicle_info=vehicle_info, image_path=image_path)

@app.route('/delete_by_source', methods=['POST'])
def delete_by_source():
    source = request.form.get('source')
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM plates WHERE source = ?", (source,))
    conn.commit()
    conn.close()
    flash(f"✅ Toutes les plaques de source {source} ont été supprimées.")
    return redirect(url_for('admin'))

@app.route('/delete_selected', methods=['POST'])
def delete_selected():
    ids = request.form.getlist('selected_ids')
    if ids:
        conn = get_connection()
        cur = conn.cursor()
        cur.executemany("DELETE FROM plates WHERE id = ?", [(id,) for id in ids])
        conn.commit()
        conn.close()
        flash(f"✅ {len(ids)} plaque(s) supprimée(s) avec succès.")
    else:
        flash("❌ Aucune plaque sélectionnée.")
    return redirect(url_for('admin'))

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r

import os

@app.route('/clear_uploads')
def clear_uploads():
    folder = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Erreur lors de la suppression de {file_path}: {e}')
    return "Uploads vidés avec succès."

@app.route('/manual_select', methods=['POST'])
def manual_select():
    image_path = request.form['image_path']
    # Nettoyage du chemin pour éviter les doublons 'exports/'
    if image_path.startswith('exports/'):
        image_path = image_path.replace('exports/', '')
    return render_template('manual_select.html', image_path=image_path)


@app.route('/submit_selection', methods=['GET','POST'])
def submit_selection():
    try:
        # 1. Récupérer TOUS les champs du formulaire dès le début (évite UnboundLocalError)
        image_path = request.form['image_path']  # <-- Maintenant garantie d'exister
        x = int(request.form['x'])
        y = int(request.form['y'])
        width = int(request.form['width'])
        height = int(request.form['height'])
        
        # 2. Validation minimale de la sélection
        if width < 50 or height < 20:
            flash("La zone sélectionnée est trop petite (min. 50x20px)", "warning")
            return redirect(url_for('manual_select', image_path=image_path))

        # 3. Vérifier que l'image existe
        img_path = os.path.join('static', 'exports', image_path)
        if not os.path.exists(img_path):
            flash("Image introuvable", "danger")
            return redirect(url_for('index'))

        # 4. Extraire la région sélectionnée (ROI)
        img = cv2.imread(img_path)
        roi = img[y:y+height, x:x+width]

        # 5. Pré-traitement pour l'OCR
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 6. Configuration Tesseract optimisée pour plaques
        config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(binary, config=config).strip()

        # 7. Nettoyage du texte
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        if not cleaned or len(cleaned) < 4:
            flash("Aucune plaque valide détectée dans la zone sélectionnée", "warning")
            return redirect(url_for('manual_select', image_path=image_path))

        # 8. Sauvegarde en base de données
        save_plate(cleaned, source='manual', image_path=f"exports/{image_path}")

        # 9. Redirection vers les résultats
        return redirect(url_for('result', 
                             media_type='image',
                             plates=[cleaned],
                             video_name=image_path))

    except KeyError as e:
        # Gère l'absence de champs dans le formulaire
        flash(f"Erreur: Champ manquant ({str(e)})", "danger")
        return redirect(url_for('index'))

    except Exception as e:
        # Gère toutes les autres erreurs avec image_path désormais accessible
        flash(f"Erreur lors de la sélection : {str(e)}", "danger")
        return redirect(url_for('manual_select', image_path=image_path))
    
    
@app.route('/retry_ocr/<path:image_path>')
def retry_ocr(image_path):
    img = cv2.imread(os.path.join('static', 'exports', image_path))
    if img is None:
        flash("❌ Image non trouvée", "danger")
        return redirect(url_for('history'))
    
    _, cleaned_text = extract_text(img, bbox=None, pad=10)
    
    if cleaned_text and cleaned_text.upper() not in ['NO NUMBER', 'NO TEXT']:
        save_plate(cleaned_text, source='retry', image_path=f"exports/{image_path}")

    return redirect(url_for('result', 
                         media_type='image',
                         plates=[cleaned_text] if cleaned_text else ['Aucune lecture OCR'],
                         video_name=image_path))


@app.route('/reprocess_image', methods=['POST'])
def reprocess_image():
    try:
        image_path = request.form['image_path']
        full_path = os.path.join('static', 'exports', image_path)
        img = cv2.imread(full_path)
        
        if img is None:
            flash("❌ Image non trouvée", "danger")
            return redirect(url_for('history'))

        # Essais progressifs
        results = []
        for pad in [0, 2, 5]:  # Différents paddings
            _, text = extract_text(img, bbox=None, pad=pad)
            if text and text not in ['Aucune lecture OCR', 'NO NUMBER']:
                results.append(text)

        # Prend le résultat le plus long (le plus probable)
        final_text = max(results, key=len) if results else 'Aucune lecture OCR'

        return redirect(url_for('result', 
                            media_type='image',
                            plates=[final_text],
                            video_name=image_path))
                            
    except Exception as e:
        print(f"Erreur reprocess: {str(e)}")
        flash("❌ Erreur lors du retraitement", "danger")
        return redirect(url_for('index'))

@app.route('/enhance_image', methods=['POST'])
def enhance_image():
    try:
        image_path = request.form['image_path']  # Doit être présent dans le formulaire
        full_path = os.path.join('static', 'exports', image_path)
        
        if not os.path.exists(full_path):
            flash("Image introuvable", "danger")
            return redirect(url_for('history'))

        # Charger et traiter l'image
        img = cv2.imread(full_path)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        enhanced = cv2.merge((l_enhanced, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)

        # Sauvegarder
        enhanced_path = os.path.join('static', 'exports', f"enhanced_{image_path}")
        cv2.imwrite(enhanced_path, enhanced)
        
        return redirect(url_for('result', 
                            media_type='image',
                            plates=["Qualité améliorée"],
                            video_name=f"enhanced_{image_path}"))

    except Exception as e:
        flash(f"Erreur lors de l'amélioration : {str(e)}", "danger")
        return redirect(url_for('history'))
            
from database import init_db
init_db()

if __name__ == '__main__':
    app.run(debug=True)