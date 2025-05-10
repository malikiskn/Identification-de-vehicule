import sqlite3
from datetime import datetime
import os

# Connexion à la base (défaut = detections.db dans ce dossier)
def get_connection(db_name='detections.db'):
    db_path = os.path.join(os.path.dirname(__file__), db_name)
    return sqlite3.connect(db_path)

def save_plate(plate, source='image', image_path=None):
    try:
        from datetime import datetime
        conn = get_connection()
        cur = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cur.execute("""
            INSERT INTO plates (plate, source, timestamp, image_path)
            VALUES (?, ?, ?, ?)
        """, (plate, source, timestamp, image_path))
        conn.commit()
        conn.close()
    except sqlite3.OperationalError as e:
        print("⚠️ Base verrouillée (save_plate) :", e)

# Récupération des plaques
def get_all_plates(db_name='detections.db'):
    conn = get_connection(db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM plates ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

#modification de la plaque
def update_plate(plate_id, new_plate):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE plates SET plate = ? WHERE id = ?", (new_plate, plate_id))
    conn.commit()
    conn.close()


def get_connection(db_name='detections.db'):
    import sqlite3
    return sqlite3.connect(db_name, timeout=10)


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS plates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT,
            source TEXT,
            timestamp TEXT,
            image_path TEXT
        )
    """)
    conn.commit()
    conn.close()

