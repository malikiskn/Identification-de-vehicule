## Malick Sokhona 21910279
## Rousseau Pierre-alexandre 21902963

# Système d'Identification de Véhicules par Détection de Plaques d'Immatriculation

## 🚗 Description

Un système complet de reconnaissance automatique de plaques d'immatriculation (ALPR) comprenant :
- Détection des véhicules avec YOLOv5
- Reconnaissance optique de caractères (OCR) pour les plaques
- Interface web interactive avec Flask
- Historique des détections dans une base de données

## 🔐 Accès Administration

L'interface d'administration est protégée par mot de passe.  
**Identifiants par défaut :**  
- URL : `/admin`  
- Mot de passe : `admin`  

Pour modifier le mot de passe, éditez la variable `ADMIN_PASSWORD` dans `app.py`.

## 🛠️ Technologies utilisées

| Composant        | Technologie                          |
|------------------|--------------------------------------|
| Modèle de détection | YOLOv5 (exporté en ONNX)            |
| Backend          | Flask (Python)                       |
| Traitement d'image | OpenCV, PyTesseract (OCR)           |
| Base de données  | SQLite                               |
| Frontend         | HTML5, CSS3, Bootstrap 5             |
| Déploiement      | Peut être conteneurisé avec Docker   |

## 📸 Fonctionnalités clés

- **Détection en temps réel** via webcam
- **Traitement par lots** d'images/vidéos
- **Correction manuelle** des détections
- **Export des résultats** (CSV, PDF)
- **Tableau de bord** d'administration

## 🚀 Installation et Utilisation

### Prérequis
- Python 3.8+
- Tesseract OCR (installation système requise)

### Configuration
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
 