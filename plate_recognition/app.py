from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import cv2
from yolo_pipeline import yolo_predictions
from database import save_plate
from datetime import datetime
from config import INPUT_WIDTH, INPUT_HEIGHT
from database import get_connection
import io
from flask import send_file
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import csv
from flask import Response
import time 
app = Flask(__name__)

app.secret_key = "supersecretkey"
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
STATIC_FOLDER = os.path.join(os.path.dirname(__file__), 'static')
RESULT_IMG_PATH = os.path.join(STATIC_FOLDER, 'result.jpg')
RAW_VIDEO_PATH = os.path.join(STATIC_FOLDER, 'raw_video.avi')  # temporaire
RESULT_VIDEO_PATH = os.path.join(STATIC_FOLDER, 'result_video.mp4')  # final
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'MP4'}
EXPORT_FOLDER = os.path.join(STATIC_FOLDER, 'exports')
os.makedirs(EXPORT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


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


# 🔹 Accueil
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    file = request.files['file']
    if not allowed_file(file.filename):
        return "❌ Format non supporté."

    filename = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    image = cv2.imread(file_path)

    net = cv2.dnn.readNetFromONNX('../runs/train/Model/weights/best.onnx')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    result_img, texts = yolo_predictions(image, net)

    # ✅ Nettoyage & récupération de la première plaque
    from datetime import datetime
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    cleaned_plate = None
    for plate in texts:
        if plate and plate != 'no number':
            cleaned_plate = plate.replace(" ", "").replace("-", "").upper()
            break

    # ✅ Construction du nom de fichier
    if cleaned_plate:
        image_filename = f"{cleaned_plate}_{now_str}.jpg"
    else:
        image_filename = f"image_{now_str}.jpg"

    # ✅ Sauvegarde dans /static/exports
    image_path = os.path.join(EXPORT_FOLDER, image_filename)
    cv2.imwrite(image_path, result_img)

    # ✅ Enregistrement en base
    for plate in texts:
        if plate and plate != 'no number':
            save_plate(plate, source='image', image_path=f"exports/{image_filename}")

    # ✅ Sauvegarde pour affichage dans result.html
    cv2.imwrite(RESULT_IMG_PATH, result_img)

    return redirect(url_for('result', media_type='image', plates=",".join(texts)))

# 🔹 Upload de vidéo
@app.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files['file']
    if not allowed_file(file.filename):
        return "❌ Vidéo non supportée."

    filename = file.filename
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    cap = cv2.VideoCapture(input_path)
    net = cv2.dnn.readNetFromONNX('../runs/train/Model/weights/best.onnx')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    # 🔧 Fichier brut .avi (temporaire)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(RAW_VIDEO_PATH, fourcc, fps, (width, height))

    texts = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result_img, new_texts = yolo_predictions(frame, net)
        out.write(result_img)
        texts.extend(new_texts)

    cap.release()
    out.release()

     # ✅ Nettoyer les textes pour nom de fichier
    cleaned_texts = [t.replace("-", "").replace(" ", "").upper() for t in texts if t and t != 'no number']
    main_plate = cleaned_texts[0] if cleaned_texts else None

    # ✅ Nom du fichier vidéo final
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    if main_plate:
        video_name = f"{main_plate}_{now_str}.mp4"
    else:
        video_name = f"video_{now_str}.mp4"
    video_path = os.path.join('static', 'exports', video_name)

    # Convertir AVI vers vidéo finale
    os.system(f"ffmpeg -y -i {RAW_VIDEO_PATH} -vcodec libx264 -crf 23 {video_path}")

    # ❌ Supprimer le fichier temporaire .avi
    if os.path.exists(RAW_VIDEO_PATH):
        os.remove(RAW_VIDEO_PATH)

    for plate in texts:
        if plate and plate != 'no number':
            save_plate(plate, source='video', image_path=f"exports/{video_name}")

    return redirect(url_for('result', media_type='video', plates=",".join(texts), video_name=video_name))


@app.route('/use_webcam', methods=['POST'])
def use_webcam():
    from datetime import datetime

    cap = cv2.VideoCapture(0)
    net = cv2.dnn.readNetFromONNX('../runs/train/Model/weights/best.onnx')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = 15

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(RAW_VIDEO_PATH, fourcc, fps, (width, height))

    texts = []
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
        texts.extend(new_texts)
        frame_count += 1

        if not any(t and t.lower() not in ['no number', 'no text'] for t in new_texts):
            no_plate_counter += 1
        else:
            no_plate_counter = 0

        if frame_count > min_duration and no_plate_counter > no_plate_limit:
            print("🛑 Arrêt anticipé : plus de plaques détectées.")
            break

    cap.release()
    out.release()

    print("✅ Fichier temporaire écrit :", os.path.exists(RAW_VIDEO_PATH))
    if not os.path.exists(RAW_VIDEO_PATH):
        print("❌ Le fichier AVI n'a pas été généré.")

    # Nettoyage pour nommage fichier
    cleaned_texts = [t.replace("-", "").replace(" ", "").upper() for t in texts if t and t.upper() not in ['NO NUMBER', 'NO TEXT']]
    main_plate = cleaned_texts[0] if cleaned_texts else None

    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    if main_plate:
        final_name = f"{main_plate}_{now_str}.mp4"
    else:
        final_name = f"webcam_{now_str}.mp4"

    result_path = os.path.join('static', 'exports', final_name)

    # Conversion vidéo
    ffmpeg_cmd = f"ffmpeg -y -i {RAW_VIDEO_PATH} -vcodec libx264 -crf 23 {result_path}"
    print("🛠️ Exécution de FFMPEG :", ffmpeg_cmd)
    os.system(ffmpeg_cmd)

    print("🎬 Fichier converti :", result_path)
    print("📁 Fichier existe ?", os.path.exists(result_path))

    if os.path.exists(RAW_VIDEO_PATH):
        os.remove(RAW_VIDEO_PATH)
        print("🧹 Fichier temporaire supprimé.")

    # Sauvegarde en base (hors NO TEXT et NO NUMBER)
    for plate in texts:
        if plate and plate.upper() not in ['NO NUMBER', 'NO TEXT']:
            print(f"💾 Sauvegarde DB pour : {plate} -> exports/{final_name}")
            save_plate(plate, source='webcam', image_path=f"exports/{final_name}")

    # ✅ Affichage final sans fausses plaques
    valid_texts = [t for t in texts if t and t.upper() not in ['NO NUMBER', 'NO TEXT']]
    return redirect(url_for('result', media_type='video', plates=",".join(valid_texts), video_name=final_name))

# 🔹 Résultat
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
        time=time.time  # ⬅️ très important
    )

from flask import Response
import threading

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
    net = cv2.dnn.readNetFromONNX('../runs/train/Model/weights/best.onnx')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

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

    # 🔢 Compter par source pour les graphiques
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
    flash("✅ Toutes les plaques ont été supprimées.")
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


# 📝 Affiche le formulaire de modification
@app.route('/edit/<int:id>')
def edit_plate(id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM plates WHERE id = ?", (id,))
    plate = cur.fetchone()
    conn.close()

    return render_template("edit_plate.html", plate=plate)

# 💾 Enregistre la plaque modifiée
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

    flash(f"✅ Plaque modifiée : {new_plate}", "success")
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

        flash("✅ Nouvelle plaque ajoutée avec succès.")
        return redirect(url_for('admin'))

    return render_template('add_plate.html')


@app.route('/detail/<plate>')
def vehicle_detail(plate):
    from database import get_connection
    from vehicle_info import get_vehicle_details  
    import os

    # 🔎 Connexion à la base
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM plates WHERE plate=? ORDER BY timestamp DESC", (plate,))
    rows = cur.fetchall()
    conn.close()

    # 📸 Trouver une image liée
    image_path = None
    export_dir = os.path.join(app.static_folder, 'exports')
    for file in os.listdir(export_dir):
        if plate.replace(" ", "").replace("-", "").lower() in file.lower():
            image_path = 'exports/' + file
            break
    print("🔍 Image trouvée pour la plaque :", image_path)
    # ✅ Récupérer les infos de l'API simulée
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

    
from database import init_db
init_db()

if __name__ == '__main__':
    app.run(debug=True)