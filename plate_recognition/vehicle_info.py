import requests
import os
import json

def fetch_vehicle_info(plate_number, token="TokenDemo2025", country="FR"):
    """
    Récupère les informations d'un véhicule à partir de son numéro de plaque
    Retourne :
        - (dict, None) en cas de succès, où dict est le JSON retourné par l'API
        - (None, str) en cas d'erreur, où str est le message d'erreur
    """

    # Construire l'URL avec les bons paramètres
    api_url = "https://api.apiplaqueimmatriculation.com/plaque"
    params = {
        "immatriculation": plate_number,
        "token": "TokenDemo2025",
        "pays": "FR"
    }

    try:
        # Passer en GET plutôt qu'en POST
       # print(plate_number)
        response = requests.get(api_url, params=params, headers={"Accept": "application/json"}, timeout=10)
        url = f'https://api.apiplaqueimmatriculation.com/plaque?immatriculation={plate_number}&token=TokenDemo2025A&pays=FR'
        headers = { 'Accept': 'application/json' }

        response = requests.get(url, headers=headers)
        # Pour le debug : print(response.url), print(response.status_code), print(response.text)
        if  response.status_code == 200 : #True
            vehicle_info = response.json()
         
            if 'data' in vehicle_info:
                #print(vehicle_info)
                return vehicle_info, None
            else:
                return None, "Aucune donnée trouvée pour ce numéro de plaque."
        else:
            return None, f"Erreur API : {response.status_code} — {response.text}"

    except requests.exceptions.Timeout:
        return None, "La requête à l'API a expiré."
    except requests.exceptions.RequestException as e:
        return None, f"Erreur lors de la requête à l'API : {str(e)}"


def save_vehicle_info(plate_number, filename="Imatriculation_info.json"):
    """
    Utilise recupere les infos de fetch_vehicle_info pour sauvegarder Les inforamation dans un fichier json 
    """
    raw_data, error = fetch_vehicle_info(plate_number)
    if error:
        print(f"Erreur lors de la récupération de {plate_number} : {error}")
        return

    plate_cleaned = plate_number.replace("-", "").upper()

    # Vérifie que la clé "data" est bien un dictionnaire
    data_block = raw_data.get("data")
    if not isinstance(data_block, dict):
        print(f"Données 'data' invalides ou vides pour {plate_cleaned}. Type reçu : {type(data_block)}")
        return

    # Chargement du fichier existant (ou création d'un nouveau dictionnaire)
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                all_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            all_data = {}
    else:
        all_data = {}

    # Mise à jour avec les bonnes données
    all_data[plate_cleaned] = data_block

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"Données sauvegardées pour {plate_cleaned}.")


def get_local_vehicle_info(plate_number, filename="Imatriculation_info.json"):
    """
    Permet de recuperer les information local grace a la plaque d'imatriculation
    Retourne :
        - Toute les information de la plaque
        - None  Si le fichier ou l'information de la plaque  n'hesite pas 
    """
    plate_key = plate_number.replace("-", "").upper()
    if not os.path.exists(filename):
        return None

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    if isinstance(all_data, dict) and plate_key in all_data:
        return all_data[plate_key]
    else:
        return None


def get_vehicle_details(plate_number):
    """
    Permet de recuperer les information local grace a la plaque d'imatriculation mais si elle n'hesite pas fait la recherche sur l'api tout en la sauvegardent  se qui permet recuperer les information local grace a la plaque d'imatriculation
    Retourne :
        - Toute les information de la plaque
    """
    clean_plate = plate_number.upper().replace("-", "").replace(" ", "")
    print("demande info plaque")

    info = get_local_vehicle_info(clean_plate)
    if info is None:
        print(f"Aucune donnée locale pour la plaque {clean_plate}.")
        save_vehicle_info(clean_plate)
        info =  get_local_vehicle_info(clean_plate)
    print(info)
    return info