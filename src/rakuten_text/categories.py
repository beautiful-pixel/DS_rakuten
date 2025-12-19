from typing import Dict, Optional, List
import pandas as pd


# Full category names
CATEGORY_NAMES = {
    10: "Livres techniques, éducatifs, artistiques ou spirituels",
    2705: "Romans, récits et littérature",
    2280: "Journaux, magazines et revues",
    2403: "Séries & encyclopédies",
    40: "Rétro Gaming",
    50: "Accessoires & Périphériques de Jeux Vidéo",
    60: "Consoles",
    2462: "Jeux Vidéo Modernes",
    2905: "Jeux PC en Téléchargement",
    1140: "Figurine",
    1160: "Jeu de carte à collectionner",
    1180: "Jeux de rôle & figurines",
    1280: "Jouets, Figurines et Poupées",
    1281: "Jeux éducatifs & Créatifs",
    1300: "Modélisme & Drones",
    1301: "Bébé, Jeux & Loisirs",
    1302: "Sport, Loisirs & Plein Air",
    1320: "Bébé & Puériculture",
    1560: "Équipement de la maison & décoration",
    1920: "Textiles d'intérieur",
    2060: "Décoration & Éclairage",
    2582: "Jardinage, déco & extérieur",
    2583: "Piscine & équipement de piscine",
    2585: "Jardin, Bricolage & Outillage",
    1940: "Alimentation & Épicerie",
    2220: "Animaux & Accessoires",
    2522: "Fournitures de bureau & papeterie",
}


# Short category names for display (max ~30 chars)
CATEGORY_SHORT_NAMES = {
    10: "Livres techniques",
    2705: "Romans & littérature",
    2280: "Journaux & magazines",
    2403: "Séries & encyclopédies",
    40: "Rétro Gaming",
    50: "Accessoires JV",
    60: "Consoles",
    2462: "Jeux Vidéo",
    2905: "Jeux PC",
    1140: "Figurine",
    1160: "Cartes à collectionner",
    1180: "Jeux de rôle",
    1280: "Jouets & Figurines",
    1281: "Jeux éducatifs",
    1300: "Modélisme & Drones",
    1301: "Bébé & Jeux",
    1302: "Sport & Loisirs",
    1320: "Bébé & Puériculture",
    1560: "Équipement maison",
    1920: "Textiles",
    2060: "Déco & Éclairage",
    2582: "Jardinage & déco",
    2583: "Piscine",
    2585: "Jardin & Bricolage",
    1940: "Alimentation",
    2220: "Animaux",
    2522: "Fournitures bureau",
}


# Category groups for analysis
CATEGORY_GROUPS = {
    "Livres & Médias": [10, 2705, 2280, 2403],
    "Jeux Vidéo": [40, 50, 60, 2462, 2905],
    "Jouets & Loisirs": [1140, 1160, 1180, 1280, 1281, 1300],
    "Bébé & Enfant": [1301, 1320],
    "Maison & Jardin": [1560, 1920, 2060, 2582, 2583, 2585],
    "Sport & Vie": [1302, 1940, 2220, 2522],
}


def get_category_name(code: int, short: bool = False) -> str:
    mapping = CATEGORY_SHORT_NAMES if short else CATEGORY_NAMES
    return mapping.get(code, str(code))


def get_all_categories(short: bool = False) -> Dict[int, str]:
    return CATEGORY_SHORT_NAMES.copy() if short else CATEGORY_NAMES.copy()
