# vehicle_info.py
def get_vehicle_details(plate_number):
    plate_number = plate_number.upper().replace("-", "")
    return fake_vehicle_data.get(plate_number, None)# vehicle_info.py

fake_vehicle_data = {
    "EZ975MX": {
        "marque": "Citroën",
        "modèle": "C3",
        "année": 2020,
        "couleur": "Blanc",
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
        "marque": "Volkswagen",
        "modèle": "Polo",
        "année": 2022,
        "couleur": "Blanc",
        "boite": "Manuelle",
        "carburant": "Diesel"
    },
    "AA123BC": {
        "marque": "Renault",
        "modèle": "Clio IV",
        "année": 2017,
        "couleur": "Gris",
        "boite": "Automatique",
        "carburant": "Essence"
    },
    "BB456DE": {
        "marque": "Peugeot",
        "modèle": "208",
        "année": 2019,
        "couleur": "Noir",
       "boite": "Automatique",
        "carburant": "Diesel"
    },
    "DQ308TK": {
        "marque": "Citroën",
        "modèle": "C3",
        "année": 2020,
        "couleur": "Rouge",
        "boite": "Automatique",
        "carburant": "Essence"
    },
    "1FB516HD": {
        "marque": "Volkswagen",
        "modèle": "Golf",
        "année": 2021,
        "couleur": "Bleu",
        "boite": "Automatique",
        "carburant": "Essence"
    },
    "MH20EE7601": {
        "marque": "Ford",
        "modèle": "Focus",
        "année": 2018,
        "couleur": "Rouge",
        "boite": "Automatique",
        "carburant": "Diesel"
    },
    "GJ179QG": {
        "marque": "Toyota",
        "modèle": "Yaris",
        "année": 2022,
        "couleur": "Vert",
        "boite": "Automatique",
        "carburant": "Hybride"
    },
    "BS132FM": {
        "marque": "Peugeot",
        "modèle": "307",
        "année": 2004,
        "couleur": "Bleu",
        "boite": "Automatique",
        "carburant": "Diesel"
    },
    "5379MX41": {
        "marque": "Peugeot",
        "modèle": "307",
        "année": 2004,
        "couleur": "Bleu",
        "boite": "Automatique",
        "carburant": "Diesel"
    },
    "ER130VZ": {
        "marque": "Volvo",
        "modèle": "Cl IV",
        "année": 2016,
        "couleur": "Noir",
        "boite": "Manuelle",
        "carburant": "Essence"
    }

}

def get_vehicle_details(plate_number):
    clean_plate = plate_number.upper().replace("-", "").replace(" ", "")
    return fake_vehicle_data.get(clean_plate)