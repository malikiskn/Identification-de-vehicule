# vehicle_info.py

fake_vehicle_data = {
    "EZ975MX": {
        "marque": "Renault",
        "modèle": "Clio IV",
        "année": 2018,
        "couleur": "Gris",
        "boite": "Manuelle",
        "carburant": "Essence"
    },
    "AB123CD": {
        "marque": "Peugeot",
        "modèle": "208",
        "année": 2021,
        "couleur": "Bleu",
        "boite": "Automatique",
        "carburant": "Diesel"
    },
    "GJ179QG": {
        "marque": "Volswagen",
        "modèle": "polo",
        "année": 2022,
        "couleur": "blanc",
        "boite": "Manuelle",
        "carburant": "Diesel"
    }
}

def get_vehicle_details(plate_number):
    plate_number = plate_number.upper().replace("-", "")
    return fake_vehicle_data.get(plate_number, None)