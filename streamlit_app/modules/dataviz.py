from pathlib import Path
import hashlib
import streamlit as st
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from PIL import Image

from sklearn.base import BaseEstimator, TransformerMixin

# --------------------------------------------------
# Images
# --------------------------------------------------

def add_image_path(df, img_dir):
    """Ajoute les chemins d'images au DataFrame."""
    df_copy = df.copy()
    df_copy["image_path"] = (
        Path(img_dir) / ("image_" + df_copy["imageid"].astype(str) + 
                        "_product_" + df_copy["productid"].astype(str) + ".jpg")
    ).astype(str)
    return df_copy

@st.cache_data(show_spinner="Calcul des hash des images...")
def generate_hash(paths):
    """Génère des hash d'images."""
    hashed_contents = []
    for path in paths:
        if path is None or not Path(path).exists():
            hashed_contents.append(None)
        else:
            with open(path, "rb") as f:
                hashed_contents.append(hashlib.sha1(f.read()).hexdigest())
    return hashed_contents

def add_image_hash(df):
    """Ajoute une colonne de hash au DataFrame."""
    df_copy = df.copy()
    if "image_path" in df_copy.columns:
        df_copy["hashed_image"] = generate_hash(df_copy["image_path"].tolist())
    return df_copy

def afficher_image(chemin, taille=150):
    """Affiche une image dans Streamlit."""
    if not chemin or not Path(chemin).exists():
        st.write("Image non disponible")
    else:
        st.image(chemin, width=taille)

def display_sample_images(df, n_images=4, n_categories=6):
    """Affiche un échantillon d'images par catégorie."""
    categories = np.sort(df["prdtypecode"].unique())[:n_categories]
    
    for cat in categories:
        st.markdown(f"**Catégorie {cat}**")
        df_cat = df[df["prdtypecode"] == cat].sample(
            min(2, len(df[df["prdtypecode"] == cat])),
            random_state=1
        )
        
        cols = st.columns(2)
        for col, (_, row) in zip(cols, df_cat.iterrows()):
            with col:
                afficher_image(row["image_path"], taille=250)

# --------------------------------------------------
# WordClouds
# --------------------------------------------------

def generate_wordclouds_by_group(df, text_col="designation", group_col="group", 
                                category_col="category", max_features=2000,group_width=300, group_height=300,
                                cat_width=100, cat_height=100):
    """
    Génère des WordClouds pour chaque groupe et pour chaque catégorie d'un groupe.
    Retourne un dict : {group: (wc_group, [(wc_cat, cat_name), ...])}
    """
    # Stopwords en plusieurs langues
    stop_words = (
        set(stopwords.words('english')) |
        set(stopwords.words('french')) |
        set(stopwords.words('german')) |
        set(stopwords.words('dutch'))
    )

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=list(stop_words),
        ngram_range=(1,2)
    )
    
    tfidf_matrix = vectorizer.fit_transform(df[text_col].fillna(''))
    tfidf = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out(),
        index=df.index
    )

    groups_dict = {}
    for gr in sorted(df[group_col].unique()):
        # WordCloud du groupe
        mask_group = df[group_col] == gr
        freqs_group = tfidf[mask_group].mean().sort_values(ascending=False)
        wc_group = WordCloud(width=group_width, height=group_height, background_color="white",max_words=150,
            max_font_size=60).generate_from_frequencies(freqs_group)

        # WordClouds des catégories dans le groupe
        wc_cats = []
        for cat in sorted(df[df[group_col]==gr][category_col].unique()):
            mask_cat = df[category_col] == cat
            freqs_cat = tfidf[mask_cat].mean().sort_values(ascending=False)
            wc_cat = WordCloud(width=cat_width, height=cat_height, background_color="white",max_words=100,
                max_font_size=50).generate_from_frequencies(freqs_cat)
            wc_cats.append((wc_cat, cat))
        
        groups_dict[gr] = (wc_group, wc_cats)

    return groups_dict

# --------------------------------------------------
# Heatmaps et statistiques
# --------------------------------------------------

def plot_keywords_heatmap_streamlit(df, keywords, categories, text_col="text", by="prdtypecode"):
    """
    Crée une heatmap des mots-clés.
    Version de ta fonction originale.
    """
    category_codes = sorted(df[by].unique())
    result = pd.DataFrame(index=keywords, columns=category_codes, dtype=float)
    
    for kw in keywords:
        pattern = fr"\b{re.escape(kw)}\b"
        contains_kw = df[text_col].astype(str).str.contains(pattern, na=False)
        freq = contains_kw.groupby(df[by]).mean()
        result.loc[kw, freq.index] = freq.values

    result_for_plot = result.copy()
    col_labels = [categories.get(code, str(code)) for code in category_codes]
    result_for_plot.columns = col_labels

    fig, ax = plt.subplots(figsize=(14, len(keywords)*0.6 + 4))
    sns.heatmap(result_for_plot.astype(float), annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title("Distribution des mots-clés par catégorie (proportion)")
    ax.set_xlabel("Catégorie")
    ax.set_ylabel("Mots-clés")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    return fig

def plot_category_distribution(df):
    """Crée un graphique de distribution des catégories."""
    cat_counts = df['category'].value_counts().sort_values()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    cat_counts.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Nombre de produits')
    ax.set_ylabel('Catégorie')
    ax.set_title('Distribution des produits par catégorie')
    plt.tight_layout()
    
    return fig


def plot_language_distribution(df, threshold=1000):
    """
    Crée un bar plot horizontal de la distribution des langues détectées
    (hors 'fr' et 'unknown'), avec regroupement des classes rares en 'autre'.
    """
    # Filtrage
    mask = (df["detected_lang_raw"] != "fr") & (df["detected_lang_raw"] != "unknown")
    ct = df.loc[mask, "detected_lang_raw"].value_counts()
    ct = ct.sort_values()

    # Regroupement des langues rares
    other = ct[ct <= threshold]
    ct = ct.drop(other.index)
    ct["autre"] = other.sum()

    # Création du graphique
    fig, ax = plt.subplots(figsize=(5, 3))

    ct.plot(
        kind="barh",
        ax=ax,
        color="steelblue"
    )

    ax.set_xlabel("Nombre de produits")
    ax.set_ylabel("Langue détectée")
    ax.set_title("Distribution des langues détectées (hors français)")

    # Rendu compact
    ax.tick_params(axis="both", labelsize=8)
    ax.title.set_fontsize(10)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)

    plt.tight_layout()
    return fig

def display_foreign_text_ratio(df):
    """
    Affiche la proportion de textes détectés comme étrangers dans Streamlit.
    """
    foreign_ratio = 1 - (
        df["detected_lang_raw"].isin(["fr", "unknown"]).sum() / len(df)
    )

    st.metric(
        label="Proportion de textes détéctés en langue étrangère",
        value=f"{foreign_ratio:.1%}"
    )


def plot_text_length_distribution(df):
    """Crée un graphique de distribution des longueurs de texte."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    if 'description_cleaned_len' in df.columns:
        ax1.hist(df['description_cleaned_len'], bins=50, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Longueur description')
        ax1.set_ylabel('Fréquence')
        ax1.set_title('Distribution longueur descriptions')
    
    if 'designation_cleaned_len' in df.columns:
        ax2.hist(df['designation_cleaned_len'], bins=50, alpha=0.7, color='salmon')
        ax2.set_xlabel('Longueur designation')
        ax2.set_ylabel('Fréquence')
        ax2.set_title('Distribution longueur designations')
    
    plt.tight_layout()
    return fig

def plot_description_analysis(df):
    """Crée les graphiques d'analyse des descriptions."""
    na_rates = df[df['description_cleaned'] == ""][['group', 'category']].value_counts() / df[['group', 'category']].value_counts()
    na_rates = na_rates.fillna(0).sort_values(ascending=False).reset_index()
    na_rates.rename(columns={0:'count'}, inplace=True)

    len_means = (
        df.groupby(['group', 'category'])["description_cleaned_len"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    sns.barplot(data=na_rates, x='count', y='category', hue='group', ax=ax1)
    ax1.set_xlabel("Proportion de produits sans description")
    ax1.set_ylabel("Catégorie")
    ax1.set_title("Produits sans description par catégorie")
    ax1.legend(title='Groupe', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    sns.barplot(data=len_means, x='description_cleaned_len', y='category', hue='group', ax=ax2)
    ax2.set_xlabel("Longueur moyenne des descriptions (caractères)")
    ax2.set_ylabel("Catégorie")
    ax2.set_title("Longueur moyenne des descriptions par catégorie")
    ax2.legend(title='Groupe', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

# --------------------------------------------------
# FEATURES EXTRACTED VISUALIZATION
# --------------------------------------------------


import numpy as np
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ============================================
# FEATURES DESCRIPTIVES 
# ============================================

def count_digits(text):
    """Compte le nombre de chiffres (0-9) dans une chaîne de caractères."""
    return sum(c.isdigit() for c in str(text))

def analyze_digit_features(df, text_col="text_cleaned", target_col="category"):
    """
    Analyse de la présence de chiffres par catégorie.
    Retourne DataFrame avec stats et figure.
    """
    df_copy = df.copy()
    
    # Créer colonne text_cleaned si elle n'existe pas
    if text_col not in df_copy.columns:
        df_copy[text_col] = (
            df_copy['designation_cleaned'].fillna('') + ' ' + 
            df_copy['description_cleaned'].fillna('')
        )
    
    # Compter les chiffres
    df_copy["nb_digits_text"] = df_copy[text_col].apply(count_digits)
    
    # Calculer statistiques par catégorie
    liste_categories = df_copy[target_col].unique()
    resultats = []
    
    for cat in sorted(liste_categories):
        sous_df = df_copy[df_copy[target_col] == cat]
        moyenne = sous_df["nb_digits_text"].mean()
        mediane = sous_df["nb_digits_text"].median()
        
        resultats.append({
            "category": cat,
            "mean_nb_digits": round(moyenne, 2),
            "median_nb_digits": round(mediane, 2),
            "min_nb_digits": sous_df["nb_digits_text"].min(),
            "max_nb_digits": sous_df["nb_digits_text"].max(),
            "nb_products": len(sous_df)
        })
    
    stats_digits = pd.DataFrame(resultats)
    stats_digits = stats_digits.sort_values("mean_nb_digits", ascending=False)
    
    # Créer visualisation
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Top 15 catégories
    top_15 = stats_digits.head(15)
    axes[0].barh(top_15["category"].astype(str), top_15["mean_nb_digits"], color='skyblue')
    axes[0].set_xlabel("Moyenne de chiffres")
    axes[0].set_title("Top 15 - Catégories les plus techniques")
    axes[0].invert_yaxis()
    
    # Distribution complète
    axes[1].bar(stats_digits["category"].astype(str), stats_digits["mean_nb_digits"], color='lightcoral')
    axes[1].set_xlabel("Catégorie")
    axes[1].set_ylabel("Moyenne de chiffres")
    axes[1].set_title("Moyenne de chiffres par catégorie (toutes)")
    axes[1].tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    
    return df_copy, stats_digits, fig

def analyze_unit_features(df, text_col="text_cleaned", target_col="category"):
    """
    Détection d'unités par catégorie.
    Retourne DataFrame avec stats et figures.
    """
    import re
    
    # Patterns d'unités (simplifiés pour performance)
    unit_patterns = {
        "cm": r"\b\d+\s*(cm|centimetre?s?|centimètre?s?)\b",
        "mm": r"\b\d+\s*(mm|millimetre?s?|millimètre?s?)\b",
        "m": r"\b\d+\s*(m|metre?s?|mètre?s?)\b",
        "kg": r"\b\d+\s*(kg|kilo|kilogramme?s?)\b",
        "g": r"\b\d+\s*(g|gramme?s?)\b",
        "ml": r"\b\d+\s*(ml|millilitres?|millilitre?)\b",
        "l": r"\b\d+\s*(l|litres?|litre?)\b",
        "inch": r'\b\d+\s*(\"|pouces?|po)\b',
        "ghz": r"\b\d+(?:\.\d+)?\s*ghz\b",
        "fps": r"\b\d+\s*fps\b",
        "go": r"\b\d+\s*(go|giga-?octets?)\b",
    }
    
    compiled_patterns = [re.compile(pattern, flags=re.IGNORECASE) for pattern in unit_patterns.values()]
    
    def detect_any_unit(text):
        """Retourne 1 si le texte contient au moins une unité."""
        text = str(text)
        for regex_pattern in compiled_patterns:
            if regex_pattern.search(text):
                return 1
        return 0
    
    df_copy = df.copy()
    
    # Créer colonne text_cleaned si elle n'existe pas
    if text_col not in df_copy.columns:
        df_copy[text_col] = (
            df_copy['designation_cleaned'].fillna('') + ' ' + 
            df_copy['description_cleaned'].fillna('')
        )
    
    # Détecter unités
    df_copy["has_any_unit"] = df_copy[text_col].apply(detect_any_unit)
    
    # Calculer statistiques par catégorie
    liste_categories = sorted(df_copy[target_col].unique())
    resultats = []
    
    for cat in liste_categories:
        sous_df = df_copy[df_copy[target_col] == cat]
        nb_produits = len(sous_df)
        
        if nb_produits == 0:
            continue
        
        taux_unites = sous_df["has_any_unit"].mean()
        pourcentage = taux_unites * 100
        
        resultats.append({
            "category": cat,
            "nb_products": nb_produits,
            "pct_products_with_unit": round(pourcentage, 2)
        })
    
    stats_units = pd.DataFrame(resultats)
    stats_units = stats_units.sort_values("pct_products_with_unit", ascending=False)
    
    # Créer visualisation
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Barres horizontales
    plot_df = stats_units.sort_values("pct_products_with_unit", ascending=True)
    axes[0].barh(plot_df["category"].astype(str), plot_df["pct_products_with_unit"], color='lightgreen')
    axes[0].set_xlabel("Pourcentage de produits avec unités (%)")
    axes[0].set_title("Présence d'unités par catégorie")
    
    # Top 10 catégories
    top_10 = stats_units.head(10)
    axes[1].bar(top_10["category"].astype(str), top_10["pct_products_with_unit"], color='orange')
    axes[1].set_xlabel("Catégorie")
    axes[1].set_ylabel("% de produits avec unités")
    axes[1].set_title("Top 10 catégories avec unités")
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    return df_copy, stats_units, fig

def analyze_combined_features(df, target_col="category"):
    """
    Analyse combinée chiffres vs unités.
    Retourne DataFrame combiné et scatter plot.
    """
    # Préparer données
    df_copy = df.copy()
    
    if 'text_cleaned' not in df_copy.columns:
        df_copy['text_cleaned'] = (
            df_copy['designation_cleaned'].fillna('') + ' ' + 
            df_copy['description_cleaned'].fillna('')
        )
    
    # Compter chiffres
    df_copy["nb_digits_text"] = df_copy["text_cleaned"].apply(count_digits)
    
    # Détecter unités (version simplifiée)
    import re
    unit_pattern = re.compile(r'\b\d+\s*(cm|mm|m|kg|g|ml|l|ghz|fps|go|inch)\b', flags=re.IGNORECASE)
    df_copy["has_any_unit"] = df_copy["text_cleaned"].apply(
        lambda x: 1 if unit_pattern.search(str(x)) else 0
    )
    
    # Calculer stats combinées
    liste_categories = sorted(df_copy[target_col].unique())
    resultats_cat = []
    
    for cat in liste_categories:
        sous_df = df_copy[df_copy[target_col] == cat]
        nb_produits = len(sous_df)
        
        if nb_produits == 0:
            continue
        
        mean_nb_digits = sous_df["nb_digits_text"].mean()
        taux_unites = sous_df["has_any_unit"].mean()
        pct_with_unit = taux_unites * 100
        
        resultats_cat.append({
            "category": cat,
            "nb_products": nb_produits,
            "mean_nb_digits": round(mean_nb_digits, 2),
            "pct_with_unit": round(pct_with_unit, 2)
        })
    
    stats_cat = pd.DataFrame(resultats_cat)
    
    # Créer scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Taille des points proportionnelle au nombre de produits
    sizes = stats_cat["nb_products"] / stats_cat["nb_products"].max() * 500
    
    scatter = ax.scatter(
        stats_cat["mean_nb_digits"],
        stats_cat["pct_with_unit"],
        s=sizes,
        alpha=0.6,
        c='steelblue',
        edgecolors='black'
    )
    
    ax.set_xlabel("Nombre moyen de chiffres (titre + description)", fontsize=12)
    ax.set_ylabel("Pourcentage de produits avec unités (%)", fontsize=12)
    ax.set_title("Catégories : technicité (chiffres) vs usage d'unités", fontsize=14)
    
    # Quadrants
    seuil_pct = 70
    seuil_digits = stats_cat["mean_nb_digits"].quantile(0.9)
    
    ax.axhline(y=seuil_pct, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=seuil_digits, color='gray', linestyle='--', alpha=0.5)
    
    # Annoter les catégories importantes
    for _, row in stats_cat.iterrows():
        x = row["mean_nb_digits"]
        y = row["pct_with_unit"]
        nom_cat = str(row["category"])
        
        if (y > seuil_pct) or (x > seuil_digits):
            ax.text(x + 0.1, y + 1, nom_cat, fontsize=9, ha='left', va='bottom')
    
    # Légende pour la taille
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=4)
    ax.legend(handles, labels, title="Nb produits (relatif)", loc='upper right')
    
    plt.tight_layout()
    
    return stats_cat, fig
    

def analyze_numerotation_features(df):
    """
    Détection des références numérotées dans les descriptions
    Cherche spécifiquement les numéros typiques de presse et littérature.
    """
    df_copy = df.copy()
    
    # Patterns plus spécifiques pour presse et littérature
    # Cherche des numéros typiques: n°1, n°2, numéro 3, tome 4, volume 5, etc.
    patterns = [
        # Format "n°" suivi de chiffres
        r'\bn[°ºo]\s*\d+\b',
        r'\bn[°ºo]?\s*\d+\s*[/\-]\s*\d+\b',  # n°1-2, n°3/4
        
        # Format "numéro" suivi de chiffres
        r'\bnum[ée]ro?\s*\d+\b',
        r'\bnum[ée]ro?\s*\d+\s*[/\-]\s*\d+\b',
        
        # Tomes et volumes
        r'\btome?\s*\d+\b',
        r'\bvol(?:ume)?\.?\s*\d+\b',
        r'\bt\.?\s*\d+\b',
        r'\bvol\.?\s*\d+\b',
        
        # Séquences numériques simples
        r'\b\d+\s*[/\-]\s*\d+\b',  # 1/2, 3-4
        r'\b\d+\s*et\s*\d+\b',     # 1 et 2
        
        # Formats spécifiques aux magazines/presse
        r'\b\d+\s*page?s?\b',      # 12 pages
        r'\bpage?\s*\d+\b',        # page 12
        
        # Formats avec parenthèses ou crochets
        r'\([nN]°?\s*\d+\)',
        r'\[[nN]°?\s*\d+\]',
        
        # Numéros spéciaux
        r'\bnum[ée]ro?\s*sp[ée]cial\b',
        r'\bsp[ée]cial\s*\d+\b',
    ]
    
    combined_mask = pd.Series(False, index=df_copy.index)
    
    for i, pattern in enumerate(patterns):
        try:
            mask = df_copy['description_cleaned'].str.contains(
                pattern, 
                case=False, 
                na=False,
                regex=True
            )
            combined_mask = combined_mask | mask
            
            # DEBUG: Afficher le nombre pour chaque pattern
            if mask.sum() > 0:
                print(f"Pattern {i} ('{pattern[:30]}...'): {mask.sum()} produits")
        except Exception as e:
            print(f"Erreur avec pattern {i}: {e}")
            continue
    
    # Recherche également dans designation_cleaned
    for pattern in patterns:
        try:
            mask_des = df_copy['designation_cleaned'].str.contains(
                pattern, 
                case=False, 
                na=False,
                regex=True
            )
            combined_mask = combined_mask | mask_des
        except:
            continue
    
    df_copy["contains_numerotation"] = combined_mask
    
    # Si toujours rien, chercher des chiffres seuls dans un contexte probable
    if df_copy["contains_numerotation"].sum() == 0:
        print("⚠️ Recherche de chiffres seuls dans un contexte...")
        
        # Cherche des motifs comme "le 3", "numéro 45", etc.
        simple_patterns = [
            r'\b(?:le|la|les|l)\s+\d+\b',
            r'\b\d+\s*(?:ème|eme|er|ère|ere)\b',
            r'\b\d{1,3}\s*(?:pages?|p\.?)\b',
        ]
        
        for pattern in simple_patterns:
            mask_simple = df_copy['description_cleaned'].str.contains(
                pattern,
                case=False,
                na=False,
                regex=True
            )
            df_copy["contains_numerotation"] = df_copy["contains_numerotation"] | mask_simple
    
    # Compter combien de produits ont été détectés
    total_detected = df_copy["contains_numerotation"].sum()
    print(f"Total produits avec numérotation détectée: {total_detected}/{len(df_copy)} ({total_detected/len(df_copy)*100:.2f}%)")
    
    # Afficher des exemples pour debug
    if total_detected > 0:
        print("\nExemples de produits détectés:")
        examples = df_copy[df_copy["contains_numerotation"]].head(3)
        for idx, row in examples.iterrows():
            desc = str(row.get('description_cleaned', ''))[:200]
            print(f"  - {desc}...")
    
    # Statistiques par catégorie
    if 'category' in df_copy.columns:
        target_col = 'category'
    elif 'prdtypecode' in df_copy.columns:
        target_col = 'prdtypecode'
    else:
        # Créer une catégorie factice
        df_copy['category_temp'] = 'all'
        target_col = 'category_temp'
    
    # Calculer les proportions
    stats_by_category = []
    for cat in df_copy[target_col].unique():
        cat_df = df_copy[df_copy[target_col] == cat]
        total_cat = len(cat_df)
        with_ref = cat_df["contains_numerotation"].sum()
        
        if total_cat > 0:
            proportion = (with_ref / total_cat) * 100
            stats_by_category.append({
                "category": cat,
                "proportion": round(proportion, 2),
                "count_with_ref": with_ref,
                "total_count": total_cat
            })
    
    stats_by_category = pd.DataFrame(stats_by_category)
    stats_by_category = stats_by_category.sort_values("proportion", ascending=False)
    
    # Créer la visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Graphique 1: Top 15 catégories
    if len(stats_by_category) > 1:
        top_n = min(15, len(stats_by_category))
        top_data = stats_by_category.head(top_n).sort_values("proportion", ascending=True)
        
        axes[0].barh(top_data["category"].astype(str), 
                    top_data["proportion"], 
                    color='#8A2BE2', alpha=0.8)
        axes[0].set_xlabel('Proportion (%)', fontsize=10)
        axes[0].set_title(f'Top {top_n} catégories avec références numérotées', 
                         fontsize=12, fontweight='bold')
        axes[0].invert_yaxis()
        
        # Ajouter les valeurs
        for i, v in enumerate(top_data["proportion"]):
            axes[0].text(v + 0.5, i, f'{v}%', va='center', fontsize=9)
    else:
        axes[0].text(0.5, 0.5, 'Données insuffisantes', 
                    ha='center', va='center', fontsize=12)
        axes[0].set_title('Distribution par catégorie')
    
    # Graphique 2: Distribution globale
    total_with_ref = df_copy["contains_numerotation"].sum()
    total_without_ref = len(df_copy) - total_with_ref
    
    colors = ['#4CAF50', '#FF9800'] if total_with_ref > 0 else ['#FF9800', '#4CAF50']
    labels = [f'Avec référence\n({total_with_ref:,})', 
             f'Sans référence\n({total_without_ref:,})']
    
    axes[1].pie([total_with_ref, total_without_ref], 
                labels=labels,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 9})
    axes[1].set_title(f'Distribution globale\n{len(df_copy):,} produits analysés', 
                     fontsize=12, fontweight='bold')
    
    plt.suptitle('Analyse des références numérotées dans presse et littérature', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return df_copy, stats_by_category, fig
def plot_keywords_distribution_by_category(df, keyword_dict=None):
    """
    Crée UNIQUEMENT le graphique en barres empilées
    Exactement comme dans le notebook.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import chain
    
    if keyword_dict is None:
        keyword_dict = get_keyword_dict()
    
    # Copier le DataFrame
    df_copy = df.copy()
    
    # 1. Créer colonne text combiné
    df_copy["text"] = (
        df_copy["designation_cleaned"].fillna("") + " " + 
        df_copy["description_cleaned"].fillna("")
    ).str.lower()
    
    # 2. Générer les colonnes de comptage
    data = {}
    
    for cat, mots in keyword_dict.items():
        pattern = '|'.join(mots)
        data[cat + "_keywords"] = df_copy["text"].str.count(pattern)
    
    # DataFrame des comptages
    keyword_counts = pd.DataFrame(data)
    
    # 3. Catégorie dominante de mots-clés par produit
    categorie_motcle_dominante = keyword_counts.idxmax(axis=1)
    
    # 4. Tableau croisé
    distribution = pd.crosstab(df_copy['category'], categorie_motcle_dominante)
    
    # 5. Palette de couleurs
    palette1 = plt.cm.tab20.colors
    palette2 = plt.cm.tab10.colors
    palette = list(chain(palette1, palette2))[:27]
    
    # 6. Graphique en barres empilées - EXACTEMENT comme notebook
    fig, ax = plt.subplots(figsize=(16, 8))
    
    bottom = np.zeros(len(distribution))
    categories_officielles = distribution.index
    
    for i, kw_cat in enumerate(distribution.columns):
        ax.bar(
            categories_officielles,
            distribution[kw_cat],
            bottom=bottom,
            label=kw_cat.replace('_keywords', ''),
            color=palette[i % len(palette)]
        )
        bottom += distribution[kw_cat].values
    
    ax.set_ylabel("Nombre de produits", fontsize=12)
    ax.set_title("Distribution des catégories de mots-clés dominantes par catégorie officielle", 
                 fontsize=14, fontweight='bold')
    
    ax.legend(
        bbox_to_anchor=(1.05, 1), 
        loc='upper left',
        title="Catégorie de mots-clés dominante",
        fontsize=9
    )
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig  

# --------------------------------------------------
# FONCTIONS MANQUANTES 
# --------------------------------------------------

def get_keyword_dict():
    """Retourne le dictionnaire de mots-clés par catégorie (EXACTO de tu notebook)."""
    return {
        "Animaux": ["chien", "chat", "animal", "compagnie", "collier"],
        "Bureau & Papeterie": ["verso", "cahier", "encre", "papier", "recto", "a5"],
        "Épicerie": ["epices", "arôme", "chocolat", "sucre", "sachet", "capsule"],
        "Puériculture": ["langer", "bavoir", "assiette", "siege", "tétine", "poussette"],
        "Vêtement Bébé & Loisirs": ["bébé", "chaussettes", "paire", "longueur", "filles","garçons"],
        "Figurines": ["figurine", "gundam", "statuette", "officiel", "marvel", "funko"],
        "Jeux de cartes": ["mtg", "oh", "rare", "vf", "carte", "magic"],
        "Jeux de rôle & Figurines": ["halloween", "figurine", "warhammer", "prince", "masque"],
        "Bricolage & Outillage": ["arrosage", "tondeuse", "aspirateur", "appareils", "outil", "coupe", "bâche"],
        "Décoration & Équipement Jardin": ["bois", "jardin", "résistant", "tente", "parasol", "aluminium"],
        "Piscine & Accessoires": ["piscine", "filtration", "pompe", "dimensions","eau", "ronde"],
        "Accessoires & Périphériques": ["nintendo", "manette", "protection", "ps4", "silicone", "câble"],
        "Consoles": ["console", "oui", "jeu", "écran", "portable", "marque", "jeux"],
        "Jeux PC en Téléchargement": ["windows", "jeu", "directx", "plus", "téléchargement", "disque", "édition"],
        "Jeux Vidéo Modernes": ["duty","jeux", "manettes", "ps3", "xbox", "kinect"],
        "Rétro Gaming": ["japonais", "import", "langue", "titres", "sous", "français"],
        "Jeux éducatifs": ["joue", "cartes", "enfants", "éducatif", "bois", "jouer"],
        "Jouets & Figurines": ["doudou", "enfants", "cadeau", "peluche", "jouet", "puzzle"],
        "Loisirs & Plein air": ["camping", "pêche", "stress", "stream", "bracelet", "trampoline"],
        "Modélisme & Drones": ["drone", "générique", "dji", "avion", "batterie", "cámera", "one"],
        "Littérature": ["monde", "ouvrage", "siècle", "roman", "livre", "histoire", "tome"],
        "Livres spécialisés": ["guide", "édition", "histoire", "art", "collection"],
        "Presse & Magazines": ["journal", "france", "illustre", "magazine", "presse", "revue"],
        "Séries & Encyclopédies": ["lot", "livres", "tomes", "volumes", "tome", "revues"],
        "Décoration & Lumières": ["led", "noël", "lumière", "lampe", "décoration", "couleur"],
        "Textiles d'intérieur": ["oreiller", "taie", "coussin", "couverture", "canapé", "cotton"],
        "Équipement Maison": ["matelas", "assise", "bois", "table", "hauteur", "mousse"]
    }

def plot_keyword_wordcloud(keyword_dict, selected_category):
    """Crée un nuage de mots pour les mots-clés d'une catégorie."""
    if selected_category not in keyword_dict:
        return None
    
    keywords = keyword_dict[selected_category]
    text = ' '.join(keywords * 3)  # Répétition pour le visuel
    
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=50
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Mots-clés discriminants: {selected_category}', fontsize=16)
    
    return fig

# ============================================
# FONCTIONS DE MODÉLISATION BASELINE
# ============================================

def evaluate_baseline_model(df, sample_size=5000):
    """
    Évalue les modèles baseline sur un échantillon.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    
    # Échantillonner les données pour performance
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df.copy()
    
    # Préparer les données
    X = df_sample.copy()
    if 'prdtypecode' in X.columns:
        y = X['prdtypecode']
    elif 'category' in X.columns:
        y = X['category']
    else:
        st.error("Pas de variable cible trouvée")
        return {}
    
    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Importer les transformers (s'ils existent)
    try:
        from modules.features import (
            TextCleaner, MergeTextTransformer, 
            KeywordFeatureTransformer
        )
    except ImportError:
        # Définir des transformers simples si le module n'existe pas
        from sklearn.base import BaseEstimator, TransformerMixin
        
        class TextCleaner(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None): return self
            def transform(self, X):
                X_copy = X.copy()
                for col in ['description_cleaned', 'designation_cleaned']:
                    if col in X_copy.columns:
                        X_copy[col] = X_copy[col].fillna('').str.lower()
                return X_copy
        
        class MergeTextTransformer(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None): return self
            def transform(self, X):
                X_copy = X.copy()
                X_copy['merged_text'] = (
                    X_copy['designation_cleaned'].fillna('') + ' ' + 
                    X_copy['description_cleaned'].fillna('')
                )
                return X_copy
        
        class KeywordFeatureTransformer(BaseEstimator, TransformerMixin):
            def __init__(self):
                self.keyword_dict = get_keyword_dict()
            
            def fit(self, X, y=None): return self
            
            def transform(self, X):
                import numpy as np
                features = []
                feature_names = []
                
                for category, keywords in self.keyword_dict.items():
                    # Compter les occurrences
                    count = X['merged_text'].apply(
                        lambda text: sum(1 for kw in keywords if f' {kw} ' in f' {text} ')
                    )
                    features.append(count.values.reshape(-1, 1))
                    feature_names.append(f"kw_{category}")
                
                import pandas as pd
                feature_array = np.hstack(features)
                return pd.DataFrame(feature_array, columns=feature_names, index=X.index)
        
        TextCleaner = TextCleaner
        MergeTextTransformer = MergeTextTransformer
        KeywordFeatureTransformer = KeywordFeatureTransformer
    
    # Définir les modèles
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "LinearSVC": LinearSVC(random_state=42, max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=20,
            random_state=42,
        ),
    }
    
    # Pipeline pour keywords
    kw_pipe = Pipeline([
        ('merger', MergeTextTransformer()),
        ('kw', KeywordFeatureTransformer()),
    ])
    
    results = {}
    
    for name, model in models.items():
        try:
            # Pipeline complet
            pipe = Pipeline([
                ('cleaner', TextCleaner()),
                ('kw', kw_pipe),
                ('scaler', StandardScaler(with_mean=False)),
                ('model', model)
            ])
            
            # Entraînement
            pipe.fit(X_train, y_train)
            
            # Prédictions
            y_pred_train = pipe.predict(X_train)
            y_pred_val = pipe.predict(X_val)
            
            # Calcul des scores
            f1_train = f1_score(y_train, y_pred_train, average='weighted')
            f1_val = f1_score(y_val, y_pred_val, average='weighted')
            
            results[name] = {
                'f1_train': round(f1_train, 3),
                'f1_val': round(f1_val, 3),
                'pipeline': pipe
            }
            
        except Exception as e:
            st.warning(f"Erreur avec {name}: {str(e)[:100]}")
            continue
    
    return results


def plot_model_comparison(results):
    """Crée un graphique de comparaison des modèles."""
    import matplotlib.pyplot as plt
    
    models = list(results.keys())
    f1_train = [results[m]['f1_train'] for m in models]
    f1_val = [results[m]['f1_val'] for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, f1_train, width, label='Train', color='skyblue')
    ax.bar(x + width/2, f1_val, width, label='Validation', color='salmon')
    
    ax.set_xlabel('Modèle')
    ax.set_ylabel('F1 Score (weighted)')
    ax.set_title('Comparaison des modèles baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Ajouter les valeurs
    for i in range(len(models)):
        ax.text(i - width/2, f1_train[i] + 0.01, f'{f1_train[i]:.3f}', 
                ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, f1_val[i] + 0.01, f'{f1_val[i]:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_feature_importance(results):
    """Affiche l'importance des features."""
    import matplotlib.pyplot as plt
    
    if not results:
        return None
    
    # Trouver le meilleur modèle
    best_model_name = max(results.items(), key=lambda x: x[1]['f1_val'])[0]
    best_pipeline = results[best_model_name]['pipeline']
    
    try:
        # Extraire les features du transformateur de keywords
        kw_transformer = best_pipeline.named_steps['kw'].named_steps['kw']
        feature_names = list(get_keyword_dict().keys())
        
        # Extraire l'importance selon le type de modèle
        model = best_pipeline.named_steps['model']
        
        if hasattr(model, 'coef_'):
            # Modèles linéaires
            coef = model.coef_
            mean_importance = np.mean(np.abs(coef), axis=0)
            
        elif hasattr(model, 'feature_importances_'):
            # Arbre de décision
            mean_importance = model.feature_importances_
            
        else:
            # Pas d'importance disponible
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Importance non disponible\npour {best_model_name}', 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Importance des features')
            return fig
        
        # Créer DataFrame
        importance_df = pd.DataFrame({
            'feature': [f"kw_{cat}" for cat in feature_names],
            'importance': mean_importance
        }).sort_values('importance', ascending=False).head(15)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(importance_df['feature'], importance_df['importance'], 
                      color='steelblue')
        ax.set_xlabel('Importance')
        ax.set_title(f'Top 15 features importantes - {best_model_name}', fontsize=14)
        ax.invert_yaxis()
        
        # Ajouter les valeurs
        for i, (idx, row) in enumerate(importance_df.iterrows()):
            ax.text(row['importance'] + 0.001, i, f'{row["importance"]:.4f}', 
                   va='center', fontsize=9)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Erreur: {str(e)[:100]}', 
               ha='center', va='center', fontsize=12)
        return fig

# ============================================
# FONCTIONS D'ENRICHISSEMENT DES FEATURES
# ============================================

def extract_length_features(df):
    """Extrait les features de longueur pour le modèle."""
    df_copy = df.copy()
    
    # Longueur en caractères
    df_copy['len_desc_chars'] = df_copy['description_cleaned'].fillna('').str.len()
    df_copy['len_desig_chars'] = df_copy['designation_cleaned'].fillna('').str.len()
    
    # Longueur en mots
    df_copy['len_desc_words'] = df_copy['description_cleaned'].fillna('').str.split().str.len()
    df_copy['len_desig_words'] = df_copy['designation_cleaned'].fillna('').str.split().str.len()
    
    return df_copy[['len_desc_chars', 'len_desig_chars', 'len_desc_words', 'len_desig_words']]


class TextLengthFeatures(BaseEstimator, TransformerMixin):
    """Transformer pour features de longueur de texte."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return extract_length_features(X)

#--------------------------------------
   # EVALUER LE MODELE ENRICHI
#--------------------------------------
def evaluate_feature_enrichment(df, sample_size=5000):
    """
    Évalue l'apport des différentes familles de features.
    Version avec f1_train et f1_val.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    
    # Échantillonner
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df.copy()
    
    # Préparer données
    X = df_sample.copy()
    if 'prdtypecode' in X.columns:
        y = X['prdtypecode']
    elif 'category' in X.columns:
        y = X['category']
    else:
        return {}
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Transformer pour keywords
    class KeywordFeatureTransformer(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.keyword_dict = get_keyword_dict()
        
        def fit(self, X, y=None): 
            return self
        
        def transform(self, X):
            import numpy as np
            features = []
            feature_names = []
            
            # Créer merged_text si nécessaire
            if 'merged_text' not in X.columns:
                X_copy = X.copy()
                X_copy['merged_text'] = (
                    X_copy['designation_cleaned'].fillna('') + ' ' + 
                    X_copy['description_cleaned'].fillna('')
                )
            else:
                X_copy = X
            
            for category, keywords in self.keyword_dict.items():
                # Compter les occurrences
                count = X_copy['merged_text'].apply(
                    lambda text: sum(1 for kw in keywords if f' {kw} ' in f' {text} ')
                )
                features.append(count.values.reshape(-1, 1))
                feature_names.append(f"kw_{category}")
            
            feature_array = np.hstack(features)
            return pd.DataFrame(feature_array, columns=feature_names, index=X.index)
    
    # Transformer pour unités
    class UnitFeatureTransformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None): 
            return self
        
        def transform(self, X):
            import re
            import numpy as np
            
            patterns = [
                (r'\b\d+\s*cm\b', 'unit_cm'),
                (r'\b\d+\s*mm\b', 'unit_mm'),
                (r'\b\d+\s*m\b', 'unit_m'),
                (r'\b\d+\s*kg\b', 'unit_kg'),
                (r'\b\d+\s*g\b', 'unit_g'),
                (r'\b\d+\s*ml\b', 'unit_ml'),
                (r'\b\d+\s*l\b', 'unit_l'),
                (r'\b\d+\s*inch\b', 'unit_inch'),
                (r'\b\d+\s*ghz\b', 'unit_ghz'),
                (r'\b\d+\s*fps\b', 'unit_fps'),
                (r'\b\d+\s*go\b', 'unit_go'),
                (r'\bn[°ºo]\s*\d+\b', 'ref_numero'),
                (r'\bnum[ée]ro?\s*\d+\b', 'ref_numero_long'),
                (r'\btome?\s*\d+\b', 'ref_tome'),
                (r'\bvol(?:ume)?\.?\s*\d+\b', 'ref_volume'),
            ]
            
            # Créer merged_text si nécessaire
            if 'merged_text' not in X.columns:
                X_copy = X.copy()
                X_copy['merged_text'] = (
                    X_copy['designation_cleaned'].fillna('') + ' ' + 
                    X_copy['description_cleaned'].fillna('')
                )
            else:
                X_copy = X
            
            features = []
            feature_names = []
            
            for pattern, name in patterns:
                has_feature = X_copy['merged_text'].str.contains(pattern, case=False, na=False).astype(int)
                features.append(has_feature.values.reshape(-1, 1))
                feature_names.append(name)
            
            feature_array = np.hstack(features)
            return pd.DataFrame(feature_array, columns=feature_names, index=X.index)
    
    # Définir les combinaisons de features
    features_config = {
        "kw seul": FeatureUnion([
            ('kw', KeywordFeatureTransformer()),
        ]),
        "kw + unités": FeatureUnion([
            ('kw', KeywordFeatureTransformer()),
            ('unit', UnitFeatureTransformer()),
        ]),
        "kw + longueur": FeatureUnion([
            ('kw', KeywordFeatureTransformer()),
            ('len', TextLengthFeatures()),
        ]),
        "kw + unités + longueur": FeatureUnion([
            ('kw', KeywordFeatureTransformer()),
            ('unit', UnitFeatureTransformer()),
            ('len', TextLengthFeatures()),
        ]),
    }
    
    results = {}
    
    for name, feature_union in features_config.items():
        try:
            # Pipeline complet
            pipe = Pipeline([
                ('features', feature_union),
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(max_iter=1000, random_state=42))
            ])
            
            # Entraînement
            pipe.fit(X_train, y_train)
            
            # Prédictions
            y_pred_train = pipe.predict(X_train)
            y_pred_val = pipe.predict(X_val)
            
            # Scores - AJOUT DE F1_TRAIN
            f1_train = f1_score(y_train, y_pred_train, average='weighted')
            f1_val = f1_score(y_val, y_pred_val, average='weighted')
            
            # Nombre de features
            try:
                n_features = pipe.named_steps['features'].transform(X_train.iloc[:1]).shape[1]
            except:
                n_features = "N/A"
            
            results[name] = {
                'f1_train': round(f1_train, 3),  # AJOUTÉ
                'f1_val': round(f1_val, 3),
                'n_features': n_features
            }
            
        except Exception as e:
            continue
    
    return results
# ============================================
# FONCTIONS D'ENRICHISSEMENT DES FEATURES - CORRIGÉES
# ============================================

def plot_feature_enrichment_comparison(results):
    """Graphique de comparaison des combinaisons de features."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not results:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Aucun résultat disponible',
                ha='center', va='center', fontsize=12)
        ax.set_title('Comparaison des combinaisons', fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        return fig
    
    configs = list(results.keys())
    f1_vals = [results[c]['f1_val'] for c in configs]
    
    # Obtener n_features si existe, sino usar placeholder
    n_features_list = []
    for c in configs:
        if 'n_features' in results[c] and results[c]['n_features'] != "N/A":
            try:
                n_features_list.append(int(results[c]['n_features']))
            except:
                n_features_list.append(len(configs) * 10)  # Valor por defecto
        else:
            # Si no hay n_features, usar un valor por defecto o el índice
            n_features_list.append(10 * (configs.index(c) + 1))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico 1: Barras de F1 Score
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(configs)]
    bars = ax1.bar(configs, f1_vals, color=colors, alpha=0.8)
    ax1.set_xlabel('Combinaison de features', fontsize=11)
    ax1.set_ylabel('F1 Score (validation)', fontsize=11)
    ax1.set_title('Performance par combinaison de features', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, max(f1_vals) * 1.15)
    
    # Añadir valores en las barras
    for bar, val in zip(bars, f1_vals):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Rotar etiquetas del eje X - FORMA CORRECTA
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Gráfico 2: Trade-off F1 vs Número de features
    for i, (config, f1, n_feat, color) in enumerate(zip(configs, f1_vals, n_features_list, colors)):
        ax2.scatter(n_feat, f1, s=200, color=color, alpha=0.8, edgecolors='black', label=config)
        ax2.text(n_feat, f1 + 0.005, config, 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Nombre de features', fontsize=11)
    ax2.set_ylabel('F1 Score', fontsize=11)
    ax2.set_title('Trade-off: Performance vs Complexité', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_improvement_delta(results):
    """Graphique d'amélioration relative par rapport au baseline."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not results or 'kw seul' not in results:
        # Crear un gráfico vacío si no hay datos
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Données insuffisantes\npour afficher l\'amélioration',
                ha='center', va='center', fontsize=12)
        ax.set_title('Amélioration relative', fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        return fig
    
    baseline_f1 = results['kw seul']['f1_val']
    configs = list(results.keys())
    
    # Excluir el baseline del gráfico de mejoras
    configs_without_baseline = [c for c in configs if c != 'kw seul']
    
    if not configs_without_baseline:
        # Si solo hay baseline, mostrar gráfico vacío
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Aucune autre combinaison\ndisponible pour comparaison',
                ha='center', va='center', fontsize=12)
        ax.set_title('Amélioration relative', fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        return fig
    
    improvements = []
    for config in configs_without_baseline:
        improvement = results[config]['f1_val'] - baseline_f1
        improvements.append(improvement)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colores según si mejora o empeora
    colors = ['#2E8B57' if imp > 0 else '#DC143C' for imp in improvements]  # Verde/Rojo
    
    bars = ax.bar(configs_without_baseline, improvements, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Combinaison de features', fontsize=11)
    ax.set_ylabel('Δ F1 Score (amélioration)', fontsize=11)
    ax.set_title(f'Amélioration relative par rapport au baseline (F1={baseline_f1:.3f})', 
                 fontsize=13, fontweight='bold', pad=20)
    
    # Línea en cero
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    # Añadir valores en las barras
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        if abs(height) < 0.001:  # Si es prácticamente cero
            continue
            
        # Determinar posición del texto
        va = 'bottom' if imp >= 0 else 'top'
        offset = 0.002 if imp >= 0 else -0.002
        color = 'darkgreen' if imp > 0 else 'darkred'
        
        ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{imp:+.3f}', 
                ha='center', va=va, fontsize=10, fontweight='bold',
                color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Configurar ticks - FORMA CORRECTA
    ax.tick_params(axis='x', rotation=45)
    ax.set_xticklabels(configs_without_baseline, ha='right')  # Usar set_xticklabels para alinear
    
    # Cuadrícula
    ax.grid(True, alpha=0.2, axis='y', linestyle='--')
    
    # Leyenda de colores
    from matplotlib.patches import Patch
    if any(imp > 0 for imp in improvements) or any(imp < 0 for imp in improvements):
        legend_elements = []
        if any(imp > 0 for imp in improvements):
            legend_elements.append(Patch(facecolor='#2E8B57', alpha=0.7, edgecolor='black', 
                                         label='Amélioration positive'))
        if any(imp < 0 for imp in improvements):
            legend_elements.append(Patch(facecolor='#DC143C', alpha=0.7, edgecolor='black', 
                                         label='Amélioration négative'))
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Ajustar límites del eje Y
    y_min, y_max = ax.get_ylim()
    margin = max(abs(min(improvements)), abs(max(improvements))) * 0.2
    ax.set_ylim(min(y_min, min(improvements) - margin), 
                max(y_max, max(improvements) + margin))
    
    # Ajustar diseño
    plt.tight_layout()
    
    return fig


def plot_feature_importance_heatmap(df):
    fig, ax = plt.subplots(figsize=(6, 3))

    sns.heatmap(
        df,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        cbar=True
    )

    ax.set_title("Importance des blocs de features par catégorie", fontsize=9)
    ax.set_xlabel("Bloc de features", fontsize=8)
    ax.set_ylabel("Catégorie", fontsize=8)

    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    return fig


def plot_rgb_histogram(hist, title=None, y_max=None):
    """
    Affiche un histogramme RGB à partir des données passées en paramètre.
    hist : list ou tuple de 3 arrays (R, G, B)
    """
    fig, ax = plt.subplots(figsize=(5, 3))

    n_images = 1
    for c, h in zip("rgb", hist):
        ax.plot(range(256), h / n_images, color=c)

    ax.set_xlabel("Niveau d'intensité", fontsize=8)
    ax.set_ylabel("Pixels par image", fontsize=8)
    ax.set_xlim(0, 255)
    if y_max:
        ax.set_ylim(0, y_max)

    ax.tick_params(axis="both", labelsize=7)

    if title:
        ax.set_title(title, fontsize=9)

    plt.tight_layout()
    return fig

def display_image(image, caption, width=250):
    """
    Affiche une image qu'elle soit en numpy ou PIL.
    """
    if isinstance(image, np.ndarray):
        st.image(image, caption=caption, width=width)
    elif isinstance(image, Image.Image):
        st.image(image, caption=caption, width=width)
    else:
        st.warning("Format d'image non reconnu")