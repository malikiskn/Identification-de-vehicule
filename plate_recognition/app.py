from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import cv2
from yolo_pipeline import yolo_predictions
from database import save_plate
from datetime import datetime
from config import INPUT_WIDTH, INPUT_HEIGHT
from database import get_connection
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


# 🔹 Upload d’image
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

    #Enregistre l'image affichée (toujours la dernière)
    cv2.imwrite(RESULT_IMG_PATH, result_img)

    #Enregistre chaque image annotée dans /static/exports avec un nom unique
    from datetime import datetime
    image_name = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    image_path = os.path.join(EXPORT_FOLDER, image_name)
    cv2.imwrite(image_path, result_img)

    for plate in texts:
        if plate and plate != 'no number':
            save_plate(plate, source='image')

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

    # 🔐 Sauvegarder chaque vidéo avec nom unique dans /static/exports
    from datetime import datetime
    video_name = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    video_path = os.path.join('static', 'exports', video_name)

    # Convertir AVI vers vidéo finale
    os.system(f"ffmpeg -y -i {RAW_VIDEO_PATH} -vcodec libx264 -crf 23 {video_path}")

    # ❌ Supprimer le fichier temporaire .avi
    if os.path.exists(RAW_VIDEO_PATH):
        os.remove(RAW_VIDEO_PATH)

    for plate in texts:
        if plate and plate != 'no number':
            save_plate(plate, source='video')

    return redirect(url_for('result', media_type='video', plates=",".join(texts), video_name=video_name))

# 🔹 Webcam en direct
@app.route('/use_webcam', methods=['POST'])
def use_webcam():
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
    max_frames = fps * 5  # 5 secondes de capture

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        result_img, new_texts = yolo_predictions(frame, net)
        out.write(result_img)
        texts.extend(new_texts)
        frame_count += 1

    cap.release()
    out.release()

    # Convertir .avi → .mp4
    os.system(f"ffmpeg -y -i {RAW_VIDEO_PATH} -vcodec libx264 -crf 23 {RESULT_VIDEO_PATH}")

    for plate in texts:
        if plate and plate != 'no number':
            save_plate(plate, source='webcam')

    return redirect(url_for('result', media_type='video', plates=",".join(texts)))

# 🔹 Résultat
@app.route('/result')
def result():
    plates = request.args.get("plates", "").split(",")
    media_type = request.args.get("media_type", "image")
    video_name = request.args.get("video_name", "result_video.mp4")  # fallback

    return render_template("result.html", media_type=media_type, plates=plates, video_name=video_name)

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
    return jsonify({"plates": live_plates})

@app.route('/webcam_live')
def webcam_live():
    return render_template('webcam_live.html')

from database import get_all_plates


@app.route('/history')
def history():
    plates = get_all_plates()
    return render_template('history.html', plates=plates)

from flask import send_file
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io

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

    count_total = len(rows)
    count_image = sum(1 for r in rows if r[2] == "image")
    count_video = sum(1 for r in rows if r[2] == "video")
    count_webcam = sum(1 for r in rows if r[2] == "webcam")

    return render_template('admin.html',
        plates=rows,
        selected_source=source,
        count_total=count_total,
        count_image=count_image,
        count_video=count_video,
        count_webcam=count_webcam
    )

@app.route('/delete/<int:id>', methods=['POST'])
def delete_plate(id):
    
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM plates WHERE id=?", (id,))
    conn.commit()
    conn.close()
    flash("✅ Plaque supprimée avec succès.")
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

ADMIN_PASSWORD = "admin123"  # ← modifie ici ton mot de passe

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

if __name__ == '__main__':
    app.run(debug=True)