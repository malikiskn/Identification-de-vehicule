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
    
    return (len(cleaned) >= min_length 
            and has_letter 
            and has_digit 
            and not cleaned.startswith('MP'))  # Filtre les faux positifs