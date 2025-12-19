import re
import html
import string
import unicodedata
from typing import Set
import regex as reg
import pandas as pd
from ftfy import fix_text
from nltk.corpus import stopwords


# Mots vides français + anglais
NLTK_STOPWORDS = set(stopwords.words("french")) | set(stopwords.words("english"))

# Ensemble étendu de ponctuation
PUNCTUATION = set(string.punctuation) | {
    "…", "'", '"', "«", "»", "•", "·", "–", "—", "‹", "›"
}

# Phrases répétitives courantes provenant de templates HTML
BOILERPLATE_PHRASES = ["li li strong", "li li", "br br", "et de"]


def clean_text(
    text,
    # Encodage & Unicode
    fix_encoding: bool = False,
    unescape_html: bool = False,
    normalize_unicode: bool = False,
    # HTML & Structure
    remove_html_tags: bool = False,
    remove_boilerplate: bool = False,
    # Transformation de casse
    lowercase: bool = False,
    # Fusions structurelles (préserver les unités sémantiques)
    merge_dimensions: bool = False,      # "22 x 11 x 2" → "22x11x2"
    merge_units: bool = False,           # "500 g" → "500g"
    merge_durations: bool = False,       # "24 h" → "24h"
    merge_age_ranges: bool = False,      # "3-5 ans" → "3_5ans"
    tag_years: bool = False,             # "1917" → "year1917"
    # Ponctuation & Caractères spéciaux
    remove_punctuation: bool = False,    # Supprimer la ponctuation isolée
    # Filtrage de tokens
    remove_stopwords: bool = False,
    remove_single_letters: bool = False,
    remove_single_digits: bool = False,
    remove_pure_punct_tokens: bool = False,
):
    # Gérer les valeurs manquantes
    if pd.isna(text) or text is None:
        return ""

    s = str(text)

    if fix_encoding:
        s = fix_text(s)

    if unescape_html:
        s = html.unescape(s)

    if normalize_unicode:
        s = unicodedata.normalize("NFC", s)

    if remove_html_tags:
        s = reg.sub(r"<[^>]+>", " ", s)

    if lowercase:
        s = s.lower()

    if merge_dimensions:
        # Triplets : "22 x 11 x 2" → "22x11x2"
        s = re.sub(r"\b(\d+)\s*x\s*(\d+)\s*x\s*(\d+)\b", r"\1x\2x\3", s, flags=re.IGNORECASE)
        # Paires : "180 x 180" → "180x180"
        s = re.sub(r"\b(\d+)\s*x\s*(\d+)\b", r"\1x\2", s, flags=re.IGNORECASE)
        # Triplets de lettres : "L x H x L" → "LxHxL"
        s = re.sub(r"\b([lh])\s*x\s*([lh])\s*x\s*([lh])\b", r"\1x\2x\3", s, flags=re.IGNORECASE)

    if merge_units:
        # Poids/volume : "500 g" → "500g"
        s = re.sub(r"\b(\d+)\s*(kg|g|mg|ml|l)\b", r"\1\2", s, flags=re.IGNORECASE)
        # Longueur : "50 cm" → "50cm"
        s = re.sub(r"\b(\d+)\s*(mm|cm|m)\b", r"\1\2", s, flags=re.IGNORECASE)
        # Stockage : "32 Go" → "32go"
        s = re.sub(r"\b(\d+)\s*(go|gb|mo|mb)\b", r"\1\2", s, flags=re.IGNORECASE)
        # Pourcentage : "100 %" → "100pct"
        s = re.sub(r"\b(\d+)\s*%\b", r"\1pct", s, flags=re.IGNORECASE)
        # Batterie : "3000 mAh" → "3000mah"
        s = re.sub(r"\b(\d+)\s*(mah|ah)\b", r"\1\2", s, flags=re.IGNORECASE)

    if merge_durations:
        # Heures : "24 h" → "24h"
        s = re.sub(r"\b(\d+)\s*(h|heures?)\b", r"\1h", s, flags=re.IGNORECASE)
        # Jours : "7 j" → "7j"
        s = re.sub(r"\b(\d+)\s*(j|jours?)\b", r"\1j", s, flags=re.IGNORECASE)
        # Mois : "12 mois" → "12mois"
        s = re.sub(r"\b(\d+)\s*mois\b", r"\1mois", s, flags=re.IGNORECASE)
        # Années : "3 ans" → "3ans"
        s = re.sub(r"\b(\d+)\s*ans?\b", r"\1ans", s, flags=re.IGNORECASE)
        # Spécial : "24h/24" → "24h24"
        s = re.sub(r"\b24\s*h\s*/\s*24\b", "24h24", s, flags=re.IGNORECASE)
        s = re.sub(r"\b7\s*j\s*/\s*7\b", "7j7", s, flags=re.IGNORECASE)

    if merge_age_ranges:
        # "0-3 ans" → "0_3ans"
        s = re.sub(r"\b(\d+)\s*-\s*(\d+)\s*ans\b", r"\1_\2ans", s, flags=re.IGNORECASE)
        # "3-5ans" → "3_5ans" (pas d'espace avant "ans")
        s = re.sub(r"\b(\d+)\s*-\s*(\d+)ans\b", r"\1_\2ans", s, flags=re.IGNORECASE)
        # "6 ans et plus" → "6plus_ans"
        s = re.sub(r"\b(\d+)\s*ans?\s*et\s*plus\b", r"\1plus_ans", s, flags=re.IGNORECASE)

    if tag_years:
        # "1917" → "year1917" (années à 4 chiffres uniquement : 18xx, 19xx, 20xx)
        s = re.sub(r"\b(18|19|20)\d{2}\b", lambda m: f" year{m.group(0)} ", s)

    if remove_punctuation:
        # Supprimer les points qui ne sont pas dans les nombres : "Hello. World" → "Hello  World" (mais garder "3.14")
        s = reg.sub(r"(?<!\d)\.(?!\d)", " ", s)
        # Supprimer les traits d'union/deux-points/etc isolés (mais garder "bien-connu")
        s = reg.sub(r"(?<!\S)-(?!\S)", " ", s)
        s = reg.sub(r"(?<!\S):(?!\S)", " ", s)
        s = reg.sub(r"(?<!\S)·(?!\S)", " ", s)
        s = reg.sub(r"(?<!\S)/(?!\S)", " ", s)
        s = reg.sub(r"(?<!\S)\+(?!\S)", " ", s)
        s = s.replace("////", " ")

    if remove_boilerplate:
        for phrase in BOILERPLATE_PHRASES:
            if phrase:
                pattern = r"\b" + re.escape(phrase) + r"\b"
                s = re.sub(pattern, " ", s, flags=re.IGNORECASE)

    # Si un filtrage de tokens est activé, nous devons diviser en tokens
    if (remove_stopwords or remove_single_letters or
        remove_single_digits or remove_pure_punct_tokens):

        tokens = s.split()
        filtered = []

        for token in tokens:
            # Filtre : mots vides
            if remove_stopwords and token.lower() in NLTK_STOPWORDS:
                continue

            # Filtre : lettres isolées
            if remove_single_letters and len(token) == 1 and token.isalpha():
                continue

            # Filtre : chiffres isolés
            if remove_single_digits and len(token) == 1 and token.isdigit():
                continue

            # Filtre : tokens de ponctuation pure
            if remove_pure_punct_tokens and token and all(ch in PUNCTUATION for ch in token):
                continue

            filtered.append(token)

        s = " ".join(filtered)


    s = reg.sub(r"\s+", " ", s).strip()

    return s



def final_text_cleaner(text):
    if pd.isna(text) or text is None:
        return ""

    s = str(text)

    # 1) Text normalization
    s = fix_text(s)
    s = html.unescape(s)
    s = unicodedata.normalize("NFC", s)

    # 2) Remove HTML tags
    s = reg.sub(r"<[^>]+>", " ", s)

    # 3) Lowercase
    s = s.lower()

    # 4) Remove dots that are not part of numbers ("hello. world" -> "hello  world", keep "3.14")
    s = reg.sub(r"(?<!\d)\.(?!\d)", " ", s)

    # 5) Remove isolated punctuation like "-" ":" "·" "/" "+" but keep things like "bien-connu", "3-5"
    s = reg.sub(r"(?<!\S)-(?!\S)", " ", s)
    s = reg.sub(r"(?<!\S):(?!\S)", " ", s)
    s = reg.sub(r"(?<!\S)·(?!\S)", " ", s)
    s = reg.sub(r"(?<!\S)/(?!\S)", " ", s)
    s = reg.sub(r"(?<!\S)\+(?!\S)", " ", s)
    s = s.replace("////", " ")

    # 6) Final whitespace normalization
    s = reg.sub(r"\s+", " ", s).strip()

    return s


def get_available_options():

    return {
        # Encodage & Unicode
        "fix_encoding": "Corriger l'encodage de texte cassé avec ftfy",
        "unescape_html": "Décoder les entités HTML (&amp; → &)",
        "normalize_unicode": "Appliquer la normalisation Unicode NFC",

        # HTML & Structure
        "remove_html_tags": "Supprimer les balises HTML <tag>contenu</tag>",
        "remove_boilerplate": "Supprimer les phrases de template communes",

        # Transformation de casse
        "lowercase": "Convertir en minuscules",

        # Fusions structurelles
        "merge_dimensions": "Fusionner les motifs de dimensions (22 x 11 → 22x11)",
        "merge_units": "Fusionner les unités numériques (500 g → 500g)",
        "merge_durations": "Fusionner les durées (24 h → 24h)",
        "merge_age_ranges": "Fusionner les tranches d'âge (3-5 ans → 3_5ans)",
        "tag_years": "Étiqueter les années à 4 chiffres (1917 → year1917)",

        # Ponctuation
        "remove_punctuation": "Supprimer les signes de ponctuation isolés",

        # Filtrage de tokens
        "remove_stopwords": "Supprimer les mots vides français/anglais",
        "remove_single_letters": "Supprimer les caractères alphabétiques isolés",
        "remove_single_digits": "Supprimer les chiffres isolés",
        "remove_pure_punct_tokens": "Supprimer les tokens composés uniquement de ponctuation",
    }


def print_available_options():
    
    options = get_available_options()
    print("Options de nettoyage disponibles :")
    print("=" * 80)
    for option, description in options.items():
        print(f"  {option:25s} : {description}")
    print("=" * 80)




##### PREPROCESSING 2 - discretisation des données numériques

import numpy as np


LABELS_DICT = {
    'volume' : (
        # volume moyen de 5 litres à 1.5 m**3
        [0, 5e3, 1.5e6, np.float32('inf')],
        ['petit volume', 'volume moyen', 'grand volume']
    ),
    'surface' : (
        # moyen de surface équivalente à 40x40 cm (plus grand que les articles de papeterie en génrale) à 2 m**2
        [0, 1600, 2e4, np.float32('inf')],
        ['petite surface', 'surface moyenne', 'grande surface']
    ),
    'length' : (
        #[0, 5, 40, 200, np.float32('inf')],
        [0, 10, 40, 350, np.float32('inf')],
        ['petite longueur', 'longueur moyenne', 'grande longueur', 'très grande longueur']
    ),
    'weight' : (
        [0, 1.5, 20, np.float32('inf')],
        ['poids leger', 'poids moyen', 'poids lourd']
    ),
    'age' : (
        [0, 3, 15, np.float32('inf')],
        ['age bébé', 'age enfant', 'age adulte']
    ),
    'memory' : (
        [0, 10, np.float32('inf')],
        ['petite mémoire', 'grande mémoire']
    ),
    'date' : (
        #[1800, 1960, 2000, 2010, 2021, np.float32('inf')],
        [1800, 1960, 2007, 2021, np.float32('inf')],
        ['date ancienne', 'date contemporaine', 'date récente', 'date future']
    ),
    'numero' : (
        [0, 20, np.float32('inf')],
        ['petit numero', 'grand numero']
    ),
    # pour mieux distinguer la catégorie Jeux éducatifs
    'card' : (
        [0, 5, np.float32('inf')],
        ['peu de cartes', 'beaucoup de cartes']
    ),
    'piece' : (
        [0, 3, np.float32('inf')],
        ['peu de pièces', 'beaucoup de pièces']
    ),
    'number' : (
        [0, 1, 2, 5, 20, 100, np.float32('inf')],
        ['zéro', 'un', 'petit nombre', 'nombre moyen', 'grand nombre', 'très grand nombre']
    ),
}

def get_label(measure, measurement_type):
    """
    permet de discrétiser les mesures en fonctions des seuils et des labels associés
    retourne le label associé à la mesure donnée en fonction de son type
    ex: get_label(5, 'age') => 'age enfant'
    """
    thresholds, labels = LABELS_DICT[measurement_type]
    # on discrétise les mesures
    for i, l in enumerate(labels):
        if measure >= thresholds[i] and measure < thresholds[i+1]:
            label = l
    return label

# unités de base utilisées : les cm, les années, les Go...
CONVERSION = {
    # longueurs
    'mm' : 0.1,
    'cm' : 1,
    'dm' : 10,
    'm' : 100,
    # surfaces
    'mm2' : 0.01,
    'cm2' : 1,
    'dm2' : 100,
    'm2' : 10**4,
    # volumes
    'mm3' : 10**-3,
    'ml' : 1,    # car 1mL = 1 cm**3
    'cm3' : 1,
    'cl' : 10,
    'dl' : 100,
    'l' : 1000,
    'dm3' : 1000,
    'm3' : 10**6,
    # poids
    'g' : 1e-3,
    'kg' : 1,
    'tonne' : 1e3,
    # age / taille
    'mois' : 1/12,
    'ans' : 1,
    # stockage (en réalité c'est x 1024 et non x 1000 mais ce niveau de précision est inutile)
    'mo' : 10**-3,
    'go' : 1,
    'to' : 10**3,
}

def convert(value, unit):
    """
    retourne la valeur dans l'unité de base
    """
    return value * CONVERSION[unit.lower()]

def to_float(value):
    """
    retourne value sous forme de float
    """
    value = value.replace(',','.')
    value = float(value)
    return value

def compute(values, unit):
    """
    retourne la valeur finale (si il y a plusieurs dimensions 5x5x10 cm => 250)
    compute(['5','5','10'], 'cm') => 250
    
    """
    computed_value = 1
    for v in values:
        computed_value = computed_value * convert(to_float(v), unit)
    return computed_value

def decor_label(label):
    """
    façon dont un label est intégré au texte
    """
    return " ["+label.replace(' ','_')+"] "

def get_decored_labels():
    """
    retourne la liste de tous les labels définis,
    peut-être utile pour ajouter à la liste des tokens spéciaux
    """
    decored_labels = []
    for (_, labels) in LABELS_DICT.values():
        for label in labels:
            decored_labels.append(decor_label(label)[1:-1])
    return decored_labels
    
def replace_volume(match):
    len_groups = len(match.groups())
    unit = match.group(len_groups)
    # on change les unités mètre(s) en m
    if len_groups == 4 and len(unit) > 1 and unit[1].lower() in ['é', 'e', 'è']:
        unit = 'm'
    value = compute(match.groups()[:-1], unit)
    label = get_label(value, 'volume')
    label = decor_label(label)
    return label

def replace_surface(match):
    len_groups = len(match.groups())
    unit = match.group(len_groups)
    # on change les unités mètre(s) en m
    if len_groups == 3 and len(unit) > 1 and unit[1].lower() in ['é', 'e', 'è']:
        unit = 'm'
    value = compute(match.groups()[:-1], unit)
    label = get_label(value, 'surface')
    label = decor_label(label)
    return label

def replace_length(match):
    len_groups = len(match.groups())
    unit = match.group(len_groups)
    # on change les unités mètre(s) en m
    if len(unit) > 1 and unit.lower()[1] in ['é', 'e', 'è']:
        unit = 'm'
    value = compute(match.groups()[:-1], unit)
    label = get_label(value, 'length')
    label = decor_label(label)
    return label

def replace_weight(match):
    len_groups = len(match.groups())
    unit = match.group(len_groups).lower()
    # on change les unités mètre(s) en m
    if len(unit) > 2:
        if unit[1] == 'r': # pour gramme(s)
            unit = 'g'
        elif unit[1] == 'i': # pour kilo, kilogramme(s)
            unit = 'kg'
        elif unit == 'tonnes':
            unit = 'tonne'
    value = compute(match.groups()[:-1], unit)
    label = get_label(value, 'weight')
    label = decor_label(label)
    return label

def replace_age(match):
    len_groups = len(match.groups())
    unit = match.group(len_groups)
    if unit.lower() == 'an':
        unit = 'ans'
    # si c'est sous forme d'intervalle de type 12-24 mois
    # on simplifie en prenant en considération juste le haut de l'intervalle, i.e. 24
    value = compute(match.groups()[-2:-1], unit)
    label = get_label(value, 'age')
    label = decor_label(label)
    return label

def replace_memory(match):
    len_groups = len(match.groups())
    unit = match.group(len_groups)
    unit = unit.lower().replace('b','o')
    value = compute(match.groups()[:-1], unit)
    label = get_label(value, 'memory')
    label = decor_label(label)
    return label

def replace_date(match):
    value = int(match.group(1))
    label = get_label(value, 'date')
    label = decor_label(label)
    return label

def replace_numero(match):
    value = int(match.group(1))
    label = get_label(value, 'numero')
    label = decor_label(label)
    return label

def replace_power(match):
    label = decor_label("puissance exp")
    return label

def replace_tension(match):
    label = decor_label("tension exp")
    return label

def replace_capacity(match):
    label = decor_label("capacite exp")
    return label

def replace_temperature(match):
    label = decor_label("temperature exp")
    return label

def replace_energy(match):
    label = decor_label("energie exp")
    return label

def replace_card(match):
    value = int(match.group(1))
    label = get_label(value, 'card')
    label = decor_label(label)
    return label

def replace_piece(match):
    value = int(match.group(1))
    label = get_label(value, 'piece')
    label = decor_label(label)
    return label

def replace_number(match):
    value = int(match.group(1))
    label = get_label(value, 'number')
    label = decor_label(label)
    return label

def replace_units(txt):
    patterns_volume = [
        r"\b(\d+[.,]?\d*)\s*[xX]\s*(\d+[.,]?\d*)\s*[xX]\s*(\d+[.,]?\d*)\s*(cm|mm|m|m[èeé]tres?)\b",
        r"\b(\d+[.,]?\d*)\s*([mcd]?m3|[mcd]?L)\b",
    ]
    patterns_surface = [
        r"\b(\d+[.,]?\d*)\s*[xX]\s*(\d+[.,]?\d*)\s*(cm|mm|m|m[èeé]tres?)\b",
        r"\b(\d+[.,]?\d*)\s*([mcd]?m2)\b",
    ]
    pattern_length = r"\b(\d+[.,]?\d*)\s*(cm|mm|m|m[èeé]tres?)\b"
    pattern_weight = r"\b(\d+[.,]?\d*)\s*(grammes?|g|kg|kilogrammes?|kilo|tonnes?)\b"
    pattern_age = r"\b(\d+-)?(\d+)\s*(mois|ans?)\b"
    pattern_memory = r"\b(\d+)\s*([gmt][ob])\b"
    pattern_date = r"\b(?:\d{2}/\d{2}/|\d{2}/)?((?:18|19|20)\d{2})\b"
    pattern_numero = r"\b(?:num[eé]ro|num|n°|n.?|no.?)\s*(\d+)\b"
    pattern_power = r"\b(\d+[.,]?\d*)\s*(cv|w|watts?)\b"
    pattern_temperature = r"\b(\d+[.,]?\d*)\s*(°[ck]?|degr[eé]s?)\b"
    pattern_energy = r"\b(\d+[.,]?\d*)\s*(kw|kilowatts?)\b"
    pattern_capacity = r"\b(\d+[.,]?\d*)\s*(m?ah)\b"
    pattern_tension = r"\b(\d+[.,]?\d*)\s*(v|volts?)\b"
    pattern_card = r"\b(\d+)\s*(cartes?)\b"
    pattern_piece = r"\b(\d+)\s*(pi[eè]ces?|pcs)\b"
    pattern_number = r"\b(\d+)\b"

    for p in patterns_volume:
        txt = re.sub(p, replace_volume, txt, flags=re.IGNORECASE)
    for p in patterns_surface:
        txt = re.sub(p, replace_surface, txt, flags=re.IGNORECASE)
    txt = re.sub(pattern_length, replace_length, txt, flags=re.IGNORECASE)
    txt = re.sub(pattern_weight, replace_weight, txt, flags=re.IGNORECASE)
    txt = re.sub(pattern_age, replace_age, txt, flags=re.IGNORECASE)
    txt = re.sub(pattern_memory, replace_memory, txt, flags=re.IGNORECASE)
    txt = re.sub(pattern_date, replace_date, txt, flags=re.IGNORECASE)
    txt = re.sub(pattern_numero, replace_numero, txt, flags=re.IGNORECASE)
    txt = re.sub(pattern_power, replace_power, txt, flags=re.IGNORECASE)
    txt = re.sub(pattern_energy, replace_energy, txt, flags=re.IGNORECASE)
    txt = re.sub(pattern_capacity, replace_capacity, txt, flags=re.IGNORECASE)
    txt = re.sub(pattern_temperature, replace_temperature, txt, flags=re.IGNORECASE)
    txt = re.sub(pattern_tension, replace_tension, txt, flags=re.IGNORECASE)
    txt = re.sub(pattern_card, replace_card, txt, flags=re.IGNORECASE)
    txt = re.sub(pattern_piece, replace_piece, txt, flags=re.IGNORECASE)
    txt = re.sub(pattern_number, replace_number, txt, flags=re.IGNORECASE)
    return txt

def text_preprocess(txt, replace_unit=True):
    txt = clean_text(txt,
        fix_encoding=True,
        unescape_html=True,
        normalize_unicode=True,
        remove_html_tags=True,
        remove_boilerplate=False)
    if replace_unit:
        txt = replace_units(txt)
    return txt

from langdetect import detect, DetectorFactory

COL_TOKENS = ["[TITRE]", "[DESC]"]

# Fixe la graine pour des résultats reproductibles
DetectorFactory.seed = 0

# Fonction de détection
def detect_lang(text):
    try:
        return detect(text)
    except:
        return "unknown"

def get_tokens(measure_types=None):
    tokens = []
    if measure_types is None:
        measure_types = LABELS_DICT.keys()
    for measure in measure_types:
        for label in LABELS_DICT[measure][1]:
            tokens.append(decor_label(label)[1:-1])
    return tokens

def complete_txt_preprocess(X):
    txt = (
        COL_TOKENS[0] + " " + X["designation"].apply(text_preprocess) + " " +
        COL_TOKENS[1] + " " + X["description"].apply(text_preprocess)
    )
    len_title = X['designation'].apply(len)
    len_desc = X['description'].fillna('').apply(len)
    detected_lang = (
        X['designation'].apply(lambda x : text_preprocess(x, replace_unit=False)) + ' ' +
        X['description'].apply(lambda x : text_preprocess(x, replace_unit=False))
    ).apply(detect_lang)
    fr_mask = detected_lang == 'fr'
    en_mask = detected_lang == 'en'
    txt_fr = np.where(fr_mask, 1, 0)
    txt_en = np.where(en_mask, 1, 0)
    txt_other = np.where((~fr_mask) & (~en_mask), 1, 0)
    technical_labels = ["puissance exp", "tension exp", "temperature exp", "energie exp", "capacite exp"]
    technical_tokens = [decor_label(l)[1:-1] for l in technical_labels]
    dimension_tokens = get_tokens(['volume', 'surface', 'length', 'weight'])       
    date_tokens = get_tokens(['date'])
    nb_tokens = [
        np.sum(np.array([np.array(txt.str.count(t[1:-1])) for t in tokens]), axis=0)
        for tokens in [technical_tokens, dimension_tokens, date_tokens]
    ]
    nb_li = X['description'].fillna("").str.count("<li>")
    # ratio de <li> pour 100 caractères
    li_ratio = (nb_li / len_desc).fillna(0) * 100
    new_X = pd.DataFrame({
        'text' : txt,
        'nb_technical_toks' : nb_tokens[0],
        'nb_dimension_toks' : nb_tokens[1],
        'nb_date_toks' : nb_tokens[2],
        'len_title' : len_title,
        'len_desc' : len_desc,
        'txt_fr' : txt_fr,
        'txt_en' : txt_en,
        'txt_other' : txt_other,
        'li_ratio' : li_ratio,
    })
    return new_X