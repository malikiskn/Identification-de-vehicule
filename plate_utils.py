def is_valid_plate(plate):
    """Version corrigée avec filtrage strict"""
    if not plate or any(x in str(plate).upper() for x in [
        'AUCUNE LECTURE OCR', 
        'NO NUMBER', 
        'NO TEXT',
        'AUCUNE PLAQUE VALIDE'
    ]):
        return False
        
    cleaned = ''.join(c for c in str(plate) if c.isalnum() or c == '-').upper()
    
    # Règles strictes :
    has_letter = any(c.isalpha() for c in cleaned)
    has_digit = any(c.isdigit() for c in cleaned)
    min_length = 7 if '-' in cleaned else 9
    
    # Filtres supplémentaires
    if (len(cleaned) < min_length or 
        not has_letter or 
        not has_digit or 
        cleaned.startswith('MP') or  # Filtre les faux positifs
        cleaned.endswith('!') or     # Filtre les artefacts
        sum(c.isdigit() for c in cleaned) < 3):  # Doit avoir au moins 3 chiffres
        return False
        
    return True

