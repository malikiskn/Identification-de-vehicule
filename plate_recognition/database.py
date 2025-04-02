import sqlite3
from datetime import datetime
import os

# Connexion à la base (défaut = detections.db dans ce dossier)
def get_connection(db_name='detections.db'):
    db_path = os.path.join(os.path.dirname(__file__), db_name)
    return sqlite3.connect(db_path)

# Sauvegarde d'une plaque
def save_plate(plate_text, source='unknown', db_name='detections.db'):
    conn = get_connection(db_name)
    cursor = conn.cursor()

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS plates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT,
            source TEXT,
            timestamp TEXT
        )
    ''')

    cursor.execute('''
        INSERT INTO plates (plate, source, timestamp)
        VALUES (?, ?, ?)
    ''', (plate_text, source, timestamp))

    conn.commit()
    conn.close()

# Récupération des plaques
def get_all_plates(db_name='detections.db'):
    conn = get_connection(db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM plates ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows