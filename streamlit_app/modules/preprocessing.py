import html
import unicodedata
import re
import pandas as pd

def clean_text(text):
    """
    Nettoie un texte en supprimant HTML, normalisant Unicode, etc.
    Version EXACTE de ta fonction.
    """
    if pd.isna(text):
        return ""
    
    text = html.unescape(text)
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-ZÀ-ÿ0-9\s]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_text_columns(df, columns):
    """
    Nettoie plusieurs colonnes de texte.
    """
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            df_copy[f"{col}_cleaned"] = df_copy[col].apply(clean_text)
            df_copy[f"{col}_cleaned_len"] = df_copy[f"{col}_cleaned"].str.len()
    return df_copy

def add_categories_to_df(df):
    """
    Ajoute les catégories lisibles au DataFrame.
    """
    CATEGORIES_CONFIG = {
        "Livres & Revues": {"Livres spécialisés": 10, "Littérature": 2705, "Presse & Magazines": 2280, "Séries & Encyclopédies": 2403},
        "Jeux Vidéo": {"Rétro Gaming": 40, "Accessoires & Périphériques": 50, "Consoles": 60, "Jeux Vidéo Modernes": 2462, "Jeux PC en Téléchargement": 2905},
        "Collection": {"Figurines": 1140, "Jeux de cartes": 1160, "Jeux de rôle & Figurines": 1180},
        "Jouets, Jeux & Loisirs": {"Jouets & Figurines": 1280, "Jeux éducatifs": 1281, "Modélisme & Drones": 1300, "Loisirs & Plein air": 1302},
        "Bébé": {"Vêtement Bébé & Loisirs": 1301, "Puériculture": 1320},
        "Maison": {"Équipement Maison": 1560, "Textiles d'intérieur": 1920, "Décoration & Lumières": 2060},
        "Jardin & Extérieur": {"Décoration & Équipement Jardin": 2582, "Piscine & Accessoires": 2583, "Bricolage & Outillage": 2585},
        "Autres": {"Épicerie": 1940, "Animaux": 2220, "Bureau & Papeterie": 2522}
    }
    
    groups_mapper = {}
    categories_mapper = {}
    for group, g_categories in CATEGORIES_CONFIG.items():
        for cat, code in g_categories.items():
            groups_mapper[code] = group
            categories_mapper[code] = cat
    
    df_copy = df.copy()
    df_copy['category'] = df_copy['prdtypecode'].replace(categories_mapper)
    df_copy['group'] = df_copy['prdtypecode'].replace(groups_mapper)
    df_copy['group_cat'] = df_copy['group'] + ' - ' + df_copy['category']
    
    return df_copy

def get_text_stats(df):
    """
    Retourne des statistiques sur le texte nettoyé.
    """
    stats = {}
    
    if 'description_cleaned_len' in df.columns:
        stats['description'] = {
            'mean': df['description_cleaned_len'].mean(),
            'median': df['description_cleaned_len'].median(),
            'min': df['description_cleaned_len'].min(),
            'max': df['description_cleaned_len'].max(),
            'missing': (df['description_cleaned'] == "").sum()
        }
    
    if 'designation_cleaned_len' in df.columns:
        stats['designation'] = {
            'mean': df['designation_cleaned_len'].mean(),
            'median': df['designation_cleaned_len'].median(),
            'min': df['designation_cleaned_len'].min(),
            'max': df['designation_cleaned_len'].max()
        }
    
    return stats