# Identification de Véhicule par Détection de Plaques d'Immatriculation

## 🚗 Description

Ce projet a pour objectif de détecter les véhicules et reconnaître leurs plaques d'immatriculation à partir d'images ou de vidéos.  
Il repose sur un modèle YOLOv5 pré-entrainé exporté en `best.onnx` et utilise Flask pour fournir une interface web simple permettant :
- Le chargement d'images ou de vidéos.
- L'affichage des résultats de détection.
- La gestion des informations détectées.

## 🛠️ Technologies utilisées

- **YOLOv5 (best.onnx)**
- **Flask** (Interface Web)
- **OpenCV** (Traitement d'image)
- **SQLite** (Stockage des détections)
- **HTML/CSS** (Templates web)

## 🚀 Installation

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/malikiskn/Identification-de-vehicule.git
   cd Identification-de-vehicule

2. **Créer un environnement virtuel (optionnel mais recommandé)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows

3. **Installer les dépendances**
pip install -r requirements.txt

4. **Lancer l’application**
cd plate_recognition
python3 app.py

Ensuite, ouvrez votre navigateur à l'adresse **affichée dans le terminal**, généralement :

- [http://127.0.0.1:5000](http://127.0.0.1:5000)
 