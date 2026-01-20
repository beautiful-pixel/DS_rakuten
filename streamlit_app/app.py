import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from matplotlib.patches import FancyBboxPatch, Arrow


# Import des modules
from modules.preprocessing import clean_text_columns, add_categories_to_df, get_text_stats
from modules.dataviz import (
    add_image_path, add_image_hash, afficher_image, 
    generate_wordclouds_by_group, plot_keywords_heatmap_streamlit,
    plot_category_distribution, plot_text_length_distribution,
    plot_language_distribution, display_foreign_text_ratio,
    plot_description_analysis, display_sample_images,   
    analyze_digit_features, analyze_unit_features, 
    analyze_combined_features, analyze_numerotation_features, 
    get_keyword_dict, plot_keyword_wordcloud, 
    plot_keywords_distribution_by_category, evaluate_baseline_model, 
    plot_model_comparison, plot_feature_importance, 
    evaluate_feature_enrichment, plot_feature_enrichment_comparison, 
    plot_improvement_delta, plot_feature_importance_heatmap,
    plot_rgb_histogram, display_image
)

from PIL import Image
import uuid

import sys
sys.path.insert(0, '../src')
sys.path.insert(0, '..')

from data import CATEGORY_NAMES
from pipeline.multimodal import FinalPipeline
import pickle
from features.text.numeric_tokens import replace_numeric_expressions



PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DEMO_DIR = PROJECT_ROOT / "data" / "demo_uploads" / "images"
IMAGE_DEMO_DIR.mkdir(parents=True, exist_ok=True)

@st.cache_data
def load_histogram_dict(path):
    with open(path, "rb") as f:
        return pickle.load(f)

hist_dict = load_histogram_dict("artifacts/rgb_histograms.pkl")

@st.cache_data
def load_monochrome_example(path="artifacts/monochrome_example.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

example = load_monochrome_example()

@st.cache_data
def load_mean_intensity(path="artifacts/mean_intensity.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

mean_intensity = load_mean_intensity()


@st.cache_data
def load_feature_importance():
    return pd.read_parquet(
        "artifacts/feature_importance_blocks.parquet"
    )

df_importance = load_feature_importance()


@st.cache_resource
def load_model():
    return FinalPipeline(str(IMAGE_DEMO_DIR))

model = load_model()


# =================================================================================================================  
# CONFIGURATION
# =================================================================================================================  
st.set_page_config(layout="wide", page_title="Projet Data Science - Rakuten")

COLORS = {
    "primary": "#003366",
    "secondary": "#f2f2f2",
    "accent": "#d9822b"
}

st.markdown(f"""
<style>
.main {{ background-color: {COLORS['secondary']}; color: {COLORS['primary']}; }}
.sidebar .sidebar-content {{ background-color: {COLORS['primary']}; color: white; }}
.stButton>button {{ background-color: {COLORS['accent']}; color: white; }}
</style>
""", unsafe_allow_html=True)

# =================================================================================================================  
# FONCTIONS DE DONNÉES
# =================================================================================================================  
@st.cache_data
def load_complete_data():
    """Charge et prépare toutes les données."""
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" 
    IMAGE_TRAIN_DIR = DATA_DIR / "raw" / "images" / "image_train"
    
    try:
        # Chargement des données
        df = pd.read_csv(DATA_DIR / "preprocessed/preprocessed.csv")

        # Ajout des images
        if IMAGE_TRAIN_DIR.exists():
            df = add_image_path(df, str(IMAGE_TRAIN_DIR))
            df = add_image_hash(df)
        else:
            df['image_path'] = None
        
        return df
        
    except Exception as e:
        st.error(f"Erreur de chargement: {e}")
        return pd.DataFrame()
    


# =================================================================================================================  
# INITIALISATION
# ==========================#=======================================================================================  
df = load_complete_data()

if df.empty:
    st.error("Données non chargées")
    st.stop()

# =================================================================================================================  
# MENU PRINCIPAL
# =================================================================================================================  
st.sidebar.title("Navigation")
menu = [
    "1. Contexte et Problématique",
    "2. Données et Volumétrie",
    "3. Préprocessing",
    "4. Exploration des données",
    "5. Feature Engineering & Modélisation Baseline",
    "6. Stratégie de modélisation",
    "7. Vectorisation ",
    "8. Transformer Français",
    "9. Transformer Multilingue",
    "10. Sélection du modèle optimal",
    "11. Essai du modèle",
    "12. Conclusions et perspectives",
]

choice = st.sidebar.radio("Sections", menu)
st.sidebar.markdown("---")
st.sidebar.info(f"{len(df):,} produits chargés")

# =================================================================================================================  
# SECTION 1: CONTEXTE ET PROBLÉMATIQUE
# =================================================================================================================  
if choice == "1. Contexte et Problématique":
    st.header("1. Contexte et Problématique")
    
    # TITRE SPÉCIFIQUE AU PROJET RAKUTEN
    st.markdown("""
    <div style='text-align: center; background: linear-gradient(135deg, #003366, #d9822b); 
                padding: 30px; border-radius: 15px; color: white; margin-bottom: 30px;'>
    <h1 style='color: white; margin-bottom: 10px; font-size: 2.5em;'>PROJET RAKUTEN</h1>
    <h3 style='color: white; opacity: 0.9; font-weight: 400;'>Classification Automatique de Produits E-commerce</h3>
    <p style='font-size: 1.1em; margin-top: 10px; opacity: 0.9;'>Challenge Data Science - ENS & Rakuten France</p>
    </div>
    """, unsafe_allow_html=True)
    #=======================================================================================   
       # CONTEXTE RÉEL
    st.subheader("Contexte Réel")
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 25px; border-radius: 10px; border-left: 5px solid #003366;'>              
        <p style='font-size: 1.05em; line-height: 1.6; margin-bottom: 25px;'>
        <strong>Rakuten France</strong>, l'une des plus grandes marketplaces européennes, 
        référence chaque jour des milliers de nouveaux produits issus de vendeurs multiples, 
        avec des descriptions textuelles et visuelles très hétérogènes.
        </p>
        
        <div style='background-color: #ffebee; padding: 20px; border-radius: 8px; margin-bottom: 20px; 
                    border-left: 4px solid #d32f2f;'>
        <h5 style='color: #d32f2f; margin-top: 0; margin-bottom: 15px;'> Problématique de Classification</h5>
        
        <div style='display: flex; align-items: flex-start; margin-bottom: 15px;'>
        <div style='background-color: #d32f2f; color: white; width: 30px; height: 30px; border-radius: 50%; 
                    display: flex; align-items: center; justify-content: center; margin-right: 10px; 
                    flex-shrink: 0; font-size: 0.9em;'></div>
        <div>
        <strong style='color: #555;'>Coût opérationnel élevé</strong><br>
        <span style='font-size: 0.95em; color: #666;'>
        Flux continu de nouveaux produits rendant la classification humaine coûteuse et peu scalable
        </span>
        </div>
        </div>
        
        <div style='display: flex; align-items: flex-start; margin-bottom: 15px;'>
        <div style='background-color: #d32f2f; color: white; width: 30px; height: 30px; border-radius: 50%; 
                    display: flex; align-items: center; justify-content: center; margin-right: 10px; 
                    flex-shrink: 0; font-size: 0.9em;'></div>
        <div>
        <strong style='color: #555;'>Complexité décisionnelle</strong><br>
        <ul style='margin: 5px 0 0 20px; padding-left: 0; color: #666; font-size: 0.9em;'>
        <li style='margin-bottom: 5px;'>Catégories aux frontières sémantiques poreuses</li>
        <li style='margin-bottom: 5px;'>Produits multi-interprétables selon leur usage</li>
        </ul>
        </div>
        </div>
        
        <div style='display: flex; align-items: flex-start; margin-bottom: 15px;'>
        <div style='background-color: #d32f2f; color: white; width: 30px; height: 30px; border-radius: 50%; 
                    display: flex; align-items: center; justify-content: center; margin-right: 10px; 
                    flex-shrink: 0; font-size: 0.9em;'></div>
        <div>
        <strong style='color: #555;'>Impact sur le time-to-market</strong><br>
        <span style='font-size: 0.95em; color: #666;'>
        Délais d'intégration prolongés affectant la disponibilité des produits
        </span>
        </div>
        </div>
        
        <div style='display: flex; align-items: flex-start;'>
        <div style='background-color: #d32f2f; color: white; width: 30px; height: 30px; border-radius: 50%; 
                    display: flex; align-items: center; justify-content: center; margin-right: 10px; 
                    flex-shrink: 0; font-size: 0.9em;'></div>
        <div>
        <strong style='color: #555;'>Inconsistences potentielles</strong><br>
        <span style='font-size: 0.95em; color: #666;'>
        Variations de classification selon les opérateurs humains
        </span>
        </div>
        </div>
        
        </div>
        
        <div style='background-color: #e3f2fd; padding: 20px; border-radius: 8px; margin-bottom: 20px; 
                    border-left: 4px solid #1976d2;'>
        <div style='display: flex; align-items: center; margin-bottom: 8px;'>
        <div style='font-size: 1.5em; margin-right: 10px;'></div>
        <h5 style='color: #1976d2; margin: 0;'>Notre Solution</h5>
        </div>
        <p style='margin: 0; color: #555; font-size: 1.05em;'>
        Automatisation de la classification produit via des modèles de Machine Learning multimodaux, 
        exploitant conjointement les données textuelles et visuelles.
        </p>
        </div>
        
        </div>
        """, unsafe_allow_html=True)
    
     #=======================================================================================  
    # OBJECTIF TECHNIQUE
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Notre Mission Technique")
    
    st.markdown("""
    <div style='background-color: #e8f5e9; padding: 30px; border-radius: 10px; 
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);'>
    
    <div style='text-align: center; margin-bottom: 30px;'>
    <div style='display: inline-block; background: linear-gradient(135deg, #003366, #d9822b); 
                color: white; padding: 15px 30px; border-radius: 25px; font-size: 1.2em; 
                font-weight: bold;'>
    Transformer données brutes → Catégorie prédite
    </div>
    </div>
    
    <div style='display: flex; justify-content: space-between; align-items: center; text-align: center;'>
    
    <div style='flex: 1; padding: 10px;'>
    <div style='background-color: white; width: 80px; height: 80px; border-radius: 50%; 
                display: flex; align-items: center; justify-content: center; margin: 0 auto 15px;
                border: 3px solid #2196F3; box-shadow: 0 4px 8px rgba(33, 150, 243, 0.2);'>
    <span style='font-size: 2em;'>📝</span>
    </div>
    <h5 style='color: #2196F3; margin: 10px 0;'>Données Textuelles</h5>
    <p style='font-size: 0.9em; color: #666;'>
    </p>
    </div>
    
    <div style='flex: 0.3; padding: 10px;'>
    <div style='font-size: 2.5em; color: #003366; font-weight: bold;'>+</div>
    </div>
    
    <div style='flex: 1; padding: 10px;'>
    <div style='background-color: white; width: 80px; height: 80px; border-radius: 50%; 
                display: flex; align-items: center; justify-content: center; margin: 0 auto 15px;
                border: 3px solid #9C27B0; box-shadow: 0 4px 8px rgba(156, 39, 176, 0.2);'>
    <span style='font-size: 2em;'>🖼️</span>
    </div>
    <h5 style='color: #9C27B0; margin: 10px 0;'>Données Visuelles</h5>
    <p style='font-size: 0.9em; color: #666;'>
    </p>
    </div>
    
    <div style='flex: 0.3; padding: 10px;'>
    <div style='font-size: 2.5em; color: #003366; font-weight: bold;'>=</div>
    </div>
    
    <div style='flex: 1; padding: 10px;'>
    <div style='background-color: white; width: 80px; height: 80px; border-radius: 50%; 
                display: flex; align-items: center; justify-content: center; margin: 0 auto 15px;
                border: 3px solid #4CAF50; box-shadow: 0 4px 8px rgba(76, 175, 80, 0.2);'>
    <span style='font-size: 2em;'>🏷️</span>
    </div>
    <h5 style='color: #4CAF50; margin: 10px 0;'>Classification précise de produits e-commerce</h5>
    <p style='font-size: 0.9em; color: #666;'>
    </p>
    </div>
    
    </div>
    </div>
    """, unsafe_allow_html=True)
  #=======================================================================================  
    # ENJEUX BUSINESS
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Enjeux Business")
    
    enjeux_cols = st.columns(3)
    
    enjeux = [
        ("", "Réduction des Coûts", "#003366", 
         "Automatiser la catégorisation diminue significativement les coûts opérationnels liés au traitement manuel."),
        ("", "Scalabilité", "#d9822b", 
         "Capacité à absorber des volumes croissants de produits sans augmentation des ressources humaines."),
        ("", "Performance Commerciale", "#4CAF50", 
         "Une classification plus précise améliore la recherche produit, l'expérience utilisateur, le SEO et les taux de conversion.")
    ]
    
    for idx, (icon, title, color, desc) in enumerate(enjeux):
        with enjeux_cols[idx]:
            st.markdown(f"""
            <div style='background-color: white; padding: 25px; border-radius: 10px; 
                        border-top: 5px solid {color}; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                        height: 280px; display: flex; flex-direction: column;'>
            <div style='display: flex; align-items: center; margin-bottom: 15px;'>
            <div style='font-size: 2.5em; margin-right: 15px; color: {color};'>{icon}</div>
            <h4 style='color: {color}; margin: 0;'>{title}</h4>
            </div>
            <p style='font-size: 0.95em; line-height: 1.6; color: #555; flex-grow: 1;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)   

    #=======================================================================================  
    # TRANSITION VERS LA SECTION SUIVANTE
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; background: linear-gradient(135deg, #003366, #d9822b); 
                padding: 30px; border-radius: 10px; color: white; position: relative; overflow: hidden;'>
    
    <div style='position: absolute; top: -50px; right: -50px; width: 150px; height: 150px; 
                background-color: rgba(255,255,255,0.1); border-radius: 50%;'></div>
    <div style='position: absolute; bottom: -50px; left: -50px; width: 150px; height: 150px; 
                background-color: rgba(255,255,255,0.1); border-radius: 50%;'></div>
    
    <h3 style='color: white; margin-bottom: 15px; position: relative; z-index: 2;'>
     Plongeons dans les données Rakuten
    </h3>
    
    <p style='font-size: 1.1em; opacity: 0.9; margin-bottom: 20px; position: relative; z-index: 2;'>
    Découvrons ensemble les 84,916 produits que notre système va apprendre à classifier
    </p>
    
    <div style='display: inline-block; background-color: white; color: #003366; 
                padding: 12px 30px; border-radius: 25px; font-weight: bold; font-size: 1.1em;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2); position: relative; z-index: 2;'>
    Explorer le Dataset →
    </div>
    
    <div style='margin-top: 20px; position: relative; z-index: 2;'>
    <span style='font-size: 1.8em; animation: bounce 2s infinite; display: inline-block;'>👇</span>
    </div>
    
    </div>
    
    <style>
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    </style>
    """, unsafe_allow_html=True)
# ===========================================================================================================
# SECTION 2: DONNÉES ET VOLUMÉTRIE
# ===========================================================================================================
elif choice == "2. Données et Volumétrie":
    st.header("2. Données et Volumétrie")

    st.subheader("I. Point de départ : données inconnues et non interprétées")

    tab1, tab2, tab3 = st.tabs([" Côté tabulaire", " Côté image", "Observations"])


        # =========================================
        # TAB 1 — CSV BRUT
        # =========================================
    with tab1:
    
        # VOLUMÉTRIE GÉNÉRALE (Ajouté ici)
        st.markdown("### Volumétrie générale du dataset")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total produits", f"{len(df):,}", help="Nombre total des observations")
        
        with col2:
            st.metric("Catégories", df['prdtypecode'].nunique(), help="27 codes numériques distincts")
        
        with col3:
            images_count = df['image_path'].notna().sum()
            st.metric("Images", f"{images_count:,}", f"{images_count/len(df)*100:.0f}%")
        
        with col4:
            missing_desc = df['description'].isna().mean()
            st.metric("Description manquantes", f"{missing_desc:.1%}")
        
        st.markdown("---")

  #---------------------------------------------------------------------------------
        #Examples données brutes

        st.markdown("####  Exemples de données brutes")

        BASE_DIR = Path(__file__).parent.parent
        DATA_DIR = BASE_DIR / "data" / "raw"

        try:
            # Charger plus d'exemples (10 lignes)
            n_samples =10
            df_raw = pd.read_csv(
                DATA_DIR / "X_train_update.csv",
                nrows=n_samples
            ).drop("Unnamed: 0", axis=1, errors="ignore")

            y_raw = pd.read_csv(
                DATA_DIR / "Y_train_CVw08PX.csv",
                nrows=n_samples
            )["prdtypecode"]

            df_raw["prdtypecode"] = y_raw.values

            # Colonnes réellement disponibles au départ
            display_df = df_raw[
                ["prdtypecode", "designation", "description", "productid", "imageid"]
            ].copy()

            # Troncature légère pour lisibilité (mais lisible)
            display_df["designation"] = display_df["designation"].apply(
                lambda x: str(x)[:80] + "…" if len(str(x)) > 80 else str(x)
            )
            display_df["description"] = display_df["description"].apply(
                lambda x: str(x)[:200] + "…" if len(str(x)) > 200 else str(x)
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                height=420,
                hide_index=False,
                column_config={
                    "prdtypecode": st.column_config.NumberColumn(
                        "prdtypecode",
                        help="Code catégorie numérique sans signification explicite"
                    ),
                    "designation": "Designation",
                    "description": "Description",
                    "productid": "Product ID",
                    "imageid": "Image ID"
                }
            )
            #  Zoom sur un produit brut (scroll vertical)
            st.markdown("####  Focus sur un produit – description")

            # Premier produit avec description sans NaN
            sample_row = df_raw[df_raw["description"].notna()].iloc[0]

            st.markdown(f"""
            **Product ID :** `{sample_row['productid']}`  

            **Image ID :** `{sample_row['imageid']}`  

            **prdtypecode :** `{sample_row['prdtypecode']}`

            **designation :** `{sample_row['designation']}`  
            """)

            # Montrer la description
            st.text_area(
                label="Description",
                value=sample_row["description"],
                height=200  # ajustable
            )

        except Exception:
            st.warning("Impossible de charger les données brutes")

    # =========================================
    # TAB 2 — IMAGES BRUTES
    # =========================================
    with tab2:
        st.markdown("####  Images associées")
        if "image_path" in df.columns and df["image_path"].notna().any():
            images_sample = df[df["image_path"].notna()].groupby("prdtypecode", as_index=False).first().head(16)

            cols = st.columns(4)

            for idx, (_, row) in enumerate(images_sample.iterrows()):
                with cols[idx % 4]:
                    st.caption(f"Image ID : {row['imageid']}")
                    afficher_image(row["image_path"], taille=100)

            st.caption(
                "Images telles que fournies dans le dataset : "
                "cadrage, arrière-plan et qualité variables"
            )
        else:
            st.info("Aucune image disponible dans ce jeu de données")
    # =========================================
    # TAB 3 — OBSERVATIONS
    # =========================================
    with tab3:
        st.markdown("###  Ce que l’on observe à ce stade")

        col1, col2, col3 = st.columns(3)

        #  LABEL
        with col1:
            st.markdown("####  Labels incompréhensibles")
            st.markdown("""
            - Les catégories sont uniquement représentées par des **codes numériques**
            - Aucun nom métier, aucune hiérarchie connue
            - Impossible de savoir ce que représente `prdtypecode = 1234`
            """)
            st.caption("Problème : impossible d'apprendre sans comprendre la cible")

        #  TEXTE
        with col2:
            st.markdown("####  Texte bruité")
            st.markdown("""
            - Descriptions **brutes et hétérogènes**
            - Présence de balises HTML, mots marketing, répétitions
            - Longueur et qualité très variables
            """)
            st.caption(" Problème : le signal est noyé dans le bruit")

        #  IMAGE
        with col3:
            st.markdown("####  Images non standardisées")
            st.markdown("""
            - Produits mal cadrés ou partiels
            - Arrière-plans et résolutions variables
            - Contenu parfois peu informatif
            """)
            st.caption(" Problème : information visuelle difficilement exploitable")

    #=======================================================================================  
    # TRANSITION VERS LA SECTION SOLUTIONS
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Contenedor centrado
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, #003366, #d9822b);
            padding: 25px;
            border-radius: 10px;
            color: white;
            text-align: center;
            max-width: 800px;
            margin: 0 auto;
        '>
        
        <h4 style='color: white; margin-bottom: 10px;'>
        Passons des données brutes aux données exploitables
        </h4>
        
        <p style='opacity: 0.9; margin-bottom: 15px;'>
        Après avoir identifié les limites des données brutes,
        nous présentons les solutions mises en place pour préparer les données.
        </p>
        
        <div style='display: inline-block; background-color: white; color: #003366; 
                    padding: 10px 25px; border-radius: 20px; font-weight: bold;
                    box-shadow: 0 3px 10px rgba(0,0,0,0.2);'>
        Découvrir les solutions →
        </div>
        
        </div>
        """, unsafe_allow_html=True)


# ====================================================================================================================
# SECTION 3: PRÉPROCESSING
# ====================================================================================================================
elif choice == "3. Préprocessing":
    st.header("3. Préprocessing")

    
    # Créer les 3 tabs
    tab1, tab2, tab3 = st.tabs(["1. Labellisation", "2. Préprocessing Texte", "3. Préprocessing Image"])
    
    # =========================================
    # TAB 1 — LABELLISATION
    # =========================================
    with tab1:
        st.markdown("### Labellisation : Comprendre les catégories")           
        # =========================================
        # WORDCLOUDS POUR COMPRENDRE LES CATÉGORIES
        # =========================================
        st.markdown("---")
        st.markdown("#### Nuages de mots par carégorie et groupe")

        # Sélection du groupe
        groups = sorted(df['group'].unique())
        selected_group = st.selectbox("Choisir un groupe à analyser", groups, key="wordcloud_group")

        if st.button(" Générer les WordClouds", key="generate_wc"):
            with st.spinner("Génération des nuages de mots..."):
                try:
                    # Filtrer les données pour le groupe sélectionné
                    df_group = df[df['group'] == selected_group]
                    
                    if len(df_group) == 0:
                        st.warning(f"Aucune donnée disponible pour le groupe {selected_group}")
                    else:
                        # Générer les WordClouds
                        wc_dict = generate_wordclouds_by_group(
                            df_group,
                            text_col="designation_cleaned",
                            group_col="group",
                            category_col="category"
                        )
                        
                        if selected_group in wc_dict:
                            wc_group, wc_cats = wc_dict[selected_group]
                            
                            # Affichage du WordCloud du groupe
                            if wc_group:
                                st.markdown(f"##### Nuage de mots du groupe : {selected_group}")
                                
                                # CORRECCIÓN: Usar to_array() en lugar de to_image()
                                # WordCloud.to_array() retorna un array numpy directamente
                                img_array = wc_group.to_array()
                                
                                # Mostrar en Streamlit
                                st.image(img_array, 
                                        caption=f"Mots les plus discriminants - {selected_group}")
                            
                            # Affichage des WordClouds par catégorie
                            if wc_cats and len(wc_cats) > 0:
                                st.markdown(f"##### Nuages de mots par catégorie dans {selected_group}")
                                
                                # Organiser en grille
                                n_categories = len(wc_cats)
                                cols_per_row = 3
                                
                                for i in range(0, n_categories, cols_per_row):
                                    cols = st.columns(min(cols_per_row, n_categories - i))
                                    row_categories = wc_cats[i:i+cols_per_row]
                                    
                                    for j, (wc_cat, cat_name) in enumerate(row_categories):
                                        with cols[j]:
                                            # CORRECCIÓN: Usar to_array() en lugar de to_image()
                                            img_cat_array = wc_cat.to_array()
                                            
                                            # Mostrar en Streamlit
                                            st.image(img_cat_array, 
                                                    caption=f"Catégorie: {cat_name}")
                                
                                st.markdown(f"**Total de {n_categories} catégories analysées dans le groupe {selected_group}**")
                            else:
                                st.info(f"Aucune catégorie trouvée dans le groupe {selected_group}")
                        else:
                            st.warning(f"Aucun WordCloud généré pour le groupe {selected_group}")
                            
                except Exception as e:
                    st.error(f"Erreur lors de la génération des WordClouds: {str(e)}")
        
 # Afficher le mapping catégories
        col1, col2 = st.columns([2, 1])
        
        with col1:
            
            # Afficher le mapping catégories - VERSION CORRECTA
            if 'group' in df.columns and 'category' in df.columns and 'prdtypecode' in df.columns:
                st.markdown("#### Mapping des catégories")
                
                # Crear el mapping con prdtypecode original
                mapping = df[['prdtypecode', 'group', 'category']].drop_duplicates().sort_values(['group', 'category', 'prdtypecode'])
                
                # Mostrar sin el índice del DataFrame
                st.dataframe(mapping.reset_index(drop=True), use_container_width=True)
                
                # Estadísticas
                st.caption(f"**{mapping['prdtypecode'].nunique()} codes numériques → {mapping['category'].nunique()} catégories → {mapping['group'].nunique()} groupes**")
        with col2:
            st.markdown("""
            <div style='text-align: center; background-color: white; padding: 20px; border-radius: 10px; 
                        border: 2px solid #d9822b;'>
            <div style='font-size: 3em; color: #d9822b;'>27</div>
            <p><strong>catégories</strong><br>→ labellisées</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='margin-top: 20px; text-align: center;'>
            <div style='font-size: 2.5em;'>8</div>
            <p><strong>Groupes logiques</strong><br>identifiés</p>
            </div>
            """, unsafe_allow_html=True)
       
    # =========================================
    # TAB 2 — PRÉPROCESSING TEXTE
    # =========================================
    with tab2:
        st.markdown("### Préprocessing du texte")
        # =========================================
        # MÉTHODOLOGIE
        # =========================================
        st.markdown("---")
        st.markdown("#### Méthodologie de nettoyage")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Transformations appliquées:**")
            st.info("""
            1. **Décodage HTML** - Entités (&amp;, &lt;, etc.)
            2. **Suppression balises** - <div>, <b>, <br>
            3. **Normalisation Unicode** - Accents, caractères spéciaux
            4. **Minuscules** - Uniformisation
            5. **Suppression ponctuation excessive**
            6. **Espacement normalisé** - Multi-espaces → espace unique
            """)
        
        # =========================================
        # EXEMPLES AVANT/APRÈS
        # =========================================
        st.markdown("---")
        st.markdown("####  Exemples avant/après nettoyage")
        # Changer couleur texte examples
        st.markdown("""
        <style>
        /* Cambiar color del texto en text_area deshabilitado */
        .stTextArea textarea[disabled] {
            color: black !important;
            -webkit-text-fill-color: black !important;
        }
        
        /* Para el fondo también si quieres */
        .stTextArea textarea[disabled] {
            background-color: #f8f9fa !important;
        }
        </style>
        """, unsafe_allow_html=True)
        # Chercher 2 exemples avec HTML
        html_examples = df[df['description'].str.contains('<', na=False)].head(2)
        
        if not html_examples.empty:
            for idx, (_, row) in enumerate(html_examples.iterrows()):
                st.markdown(f"**Exemple {idx + 1}**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Avant**")
                    st.text_area(
                        "", 
                        value=row['description'][:1500],
                        height=200,
                        disabled=True,
                        label_visibility="collapsed"
                    )
                    st.caption(f"{len(str(row['description'])):,} car.")
                
                with col2:
                    st.markdown("**Après**")
                    if 'description_cleaned' in row:
                        st.text_area(
                            "",
                            value=row['description_cleaned'][:1500],
                            height=200,
                            disabled=True,
                            label_visibility="collapsed"
                        )
                        st.caption(f"{len(str(row['description_cleaned'])):,} car.")
                    else:
                        st.warning("Non disponible")
                
                if idx == 0:
                    st.divider()
        
        else:
            st.info("Aucun HTML trouvé")
       
    # =========================================
    # TAB 3 — PRÉPROCESSING IMAGE
    # =========================================
    with tab3:
        st.markdown("### Préprocessing des images — Recadrage automatique")

        st.markdown(
            """
            <div style='background-color: #e3f2fd; padding: 15px; border-radius: 10px;'>
            <strong>Démonstration du recadrage automatique par détection de contours (CropTransformer).</strong><br>
            Les exemples ci-dessous illustrent des cas favorables et défavorables, en fonction du contraste
            entre l’objet et le fond.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        img1 = Image.open("assets/avant_recadrage.png")
        img2 = Image.open("assets/apres_recadrage.png")

        if "show_after" not in st.session_state:
            st.session_state.show_after = False

        # Colonnes symétriques : image | spacer | bouton | spacer | image
        col_left, col_spacer1, col_button, col_spacer2, col_right, col_spacer3 = st.columns(
            [2, 0.5, 1, 0.5, 2, 3]
        )

        with col_left:
            st.image(
                img1,
                caption="Avant recadrage",
                width=350
            )

        with col_button:
            st.markdown("<br><br>", unsafe_allow_html=True)  # centrage vertical
            if st.button(
                "➜",
                use_container_width=True
            ):
                st.session_state.show_after = True

        with col_right:
            if st.session_state.show_after:
                st.image(
                    img2,
                    caption="Après recadrage",
                    width=350
                )

        
# ====================================================================================================================
# SECTION 4: EXPLORATION DES DONNÉES
# ====================================================================================================================
elif choice == "4. Exploration des données":
    st.header("3. Exploration des données")
    
    # Sous-menu
    exploration_tab = st.sidebar.selectbox(
        "Type d'exploration",
        ["Distribution catégories", "Analyse textuelle", "Visualisation images", "Mots-clés discriminants"]
    )
    
    if exploration_tab == "Distribution catégories":
        st.subheader("Distribution des catégories")
        
        # Graphique de distribution
        fig = plot_category_distribution(df)
        col_center, col_right = st.columns([5, 1])

        with col_center:
            st.pyplot(fig)
        
        # Statistiques détaillées
        st.subheader("Statistiques par catégorie")
        cat_stats = df['category'].value_counts().reset_index()
        cat_stats.columns = ['Catégorie', 'Nombre de produits']
        cat_stats['Pourcentage'] = (cat_stats['Nombre de produits'] / len(df) * 100).round(2)
        st.dataframe(cat_stats, use_container_width=True)
        
        # Par langue
        st.subheader("Langues détéctées")

        fig_lang = plot_language_distribution(df, threshold=1000)

        col_left, col_center, col_right = st.columns([2, 4, 1])

        with col_left:
            display_foreign_text_ratio(df)


        with col_center:
            st.pyplot(fig_lang)


        # Distribution par groupe
        # st.subheader("Distribution par groupe")
        # group_stats = df['group'].value_counts().reset_index()
        # group_stats.columns = ['Groupe', 'Nombre de produits']
        
        # fig2, ax = plt.subplots(figsize=(10, 4))
        # ax.barh(group_stats['Groupe'], group_stats['Nombre de produits'], color='orange', alpha=0.7)
        # ax.set_xlabel('Nombre de produits')
        # ax.set_title('Distribution des produits par groupe')
        # ax.invert_yaxis()
        # plt.tight_layout()
        # st.pyplot(fig2)
    
    elif exploration_tab == "Analyse textuelle":
        st.subheader("Analyse textuelle")
        
        # Longueur des textes
        st.subheader("Distribution des longueurs de texte")
        fig = plot_text_length_distribution(df)
        st.pyplot(fig)
        
        # Analyse des descriptions
        st.subheader("Analyse approfondie des descriptions")
        with st.expander("Cliquer pour développer", expanded=True):
            fig = plot_description_analysis(df)
            st.pyplot(fig)
        
        # Nuages de mots
        st.subheader("Nuages de mots par groupe")
        
        # Sélection du groupe
        groups = sorted(df['group'].unique())
        selected_group = st.selectbox("Choisir un groupe", groups)
        
        if st.button("Générer les WordClouds"):
            # Générer les WordClouds pour le groupe sélectionné
            wc_dict = generate_wordclouds_by_group(
                df[df['group'] == selected_group],
                text_col="designation_cleaned",
                group_col="group",
                category_col="category"
            )
            
            if selected_group in wc_dict:
                wc_group, wc_cats = wc_dict[selected_group]
                
                # Affichage du WordCloud du groupe
                if wc_group:
                    st.subheader(f"WordCloud du groupe: {selected_group}")
                    st.image(wc_group.to_array(), 
                            caption=f"Mot-clés les plus fréquents dans {selected_group}",
                            use_column_width=True)
                
                # Affichage des WordClouds par catégorie
                if wc_cats:
                    st.subheader(f"WordClouds par catégorie dans {selected_group}")
                    
                    # Organiser en grille
                    n_categories = len(wc_cats)
                    cols_per_row = 3
                    
                    for i in range(0, n_categories, cols_per_row):
                        cols = st.columns(cols_per_row)
                        row_categories = wc_cats[i:i+cols_per_row]
                        
                        for j, (wc_cat, cat_name) in enumerate(row_categories):
                            with cols[j]:
                                st.image(wc_cat.to_array(), 
                                        caption=f"Catégorie: {cat_name}",
                                        use_column_width=True)
            else:
                st.warning(f"Aucun WordCloud généré pour le groupe {selected_group}")
    
    elif exploration_tab == "Visualisation images":
        st.subheader("Visualisation des images")
        
        st.markdown("""
        **Aperçu visuel du jeu de données**
        
        Ces images, affichées aléatoirement par catégorie, illustrent la diversité visuelle du jeu de données.
        
        **Observations:**
        - La plupart des images ont un fond blanc, mais ce n'est pas systématique
        - Le cadrage des objets varie : certains apparaissent très petits dans l'image
        - Qualité d'image variable (résolution, éclairage)
        """)
        
        # Paramètres d'affichage
        n_categories = st.slider("Nombre de catégories à afficher", 2, 10, 4)
        n_images = st.slider("Images par catégorie", 1, 4, 2)
        
        if st.button("Afficher des exemples d'images"):
            display_sample_images(df, n_images=n_images, n_categories=n_categories)
        
        # Exemple de problème de cadrage
        st.subheader("Problème de cadrage - Exemple")
        
        # Chercher des images avec objet petit
        if 'image_path' in df.columns:
            # Prendre quelques exemples aléatoires
            sample_df = df[df['image_path'].notna()].sample(min(3, len(df)), random_state=42)
            
            cols = st.columns(3)
            for idx, (_, row) in enumerate(sample_df.iterrows()):
                with cols[idx]:
                    st.write(f"Catégorie: {row['category']}")
                    st.write(f"Designation: {row['designation'][:50]}...")
                    afficher_image(row['image_path'], taille=200)
    
    elif exploration_tab == "Mots-clés discriminants":
        st.subheader("Distribution des mots-clés par catégorie")
        
        keywords = ["cm", "hauteur", "piscine", "drone", "bébé", "tout", "couleur", "coussin"]
        categories_dict = {gc: gc for gc in df['group_cat'].unique()[:10]}
        
        fig = plot_keywords_heatmap_streamlit(
            df, keywords, categories_dict, 
            text_col="designation_cleaned", 
            by="group_cat"
        )
        st.pyplot(fig)
        
        # Dictionnaire de mots-clés
        with st.expander("Dictionnaire de mots-clés (extrait)"):
            kw_dict = get_keyword_dict()
            categories = list(kw_dict.keys())
            
            for i in range(0, len(categories), 3):
                cols = st.columns(3)
                for j, cat in enumerate(categories[i:i+3]):
                    with cols[j]:
                        st.write(f"**{cat}**")
                        st.write(f"*{', '.join(kw_dict[cat][:3])}...*")


# ====================================================================================================================
# SECTION 5: FEATURES & MODÉLISATION BASELINE
# ====================================================================================================================

elif choice == "5. Feature Engineering & Modélisation Baseline":
    st.header("5. Feature Engineering & Modélisation Baseline")

    # NOUVEAUX ONGLETS SELON LES DIRECTIVES
    tabs = st.tabs([
        "1. Features & Baseline - Texte",  # TAB 1 : Liste des features texte + résultats F1, matrice de confusion
        "2. Features Image - Couleurs",
        "3. Features Image - Formes",
        "4. Modèle baseline Image",
    ])

    with tabs[0]:
        st.subheader(" 1. Liste des Features Textuelles & Transformations Appliquées")

        # --- PARTIE 1 : LISTE DES FEATURES ---
        st.markdown("####  Catalogue des Features Extraites du Texte")
        col_feat, col_trans = st.columns(2)
        with col_feat:
            st.markdown("**Famille de Features**")
            st.write("- **Mots-clés** (27 catégories lexicales manuelles)")
            st.write("- **Chiffres** (Compte des nombres 0-9)")
            st.write("- **Unités de mesure** (cm, kg, GHz, Go...)")
            st.write("- **Numérotation** (références presse: 'n°', 'tome', format 123/456)")
            st.write("- **Longueur du texte** (caractères et mots)")

        with col_trans:
            st.markdown("**Transformations Appliquées**")
            st.write("- Vectorisation binaire (présence/absence)")
            st.write("- Normalisation Min-Max pour les compteurs")
            st.write("- `StandardScaler` (sans centrage pour préserver la sparsité)")

        # --- PARTIE 2 : EXEMPLE CONCRET & RÉSULTATS ---
        st.markdown("---")
        st.subheader(" 2. Exemple Features")

        # A. EXEMPLE VISUEL D'UNE FEATURE (ex: Chiffres/Unités)
        with st.expander(" **Voir un exemple pertinent de feature (ex: Chiffres & Unités)**"):
            # Tu peux réutiliser ici tes fonctions existantes
            if st.button("Afficher exemple Chiffres", key="show_digits_ex"):
                try:
                    df_chiffres, stats_chiffres, fig_chiffres = analyze_digit_features(df)
                    st.pyplot(fig_chiffres)
                    st.write("**Top catégories avec le plus de chiffres:**")
                    st.dataframe(stats_chiffres.head(5))
                except Exception as e:
                    st.error(f"Erreur: {e}")
            
            if st.button("Afficher exemple Unités", key="show_units_ex"):
                try:
                    df_unités, stats_unités, fig_unités = analyze_unit_features(df)
                    st.pyplot(fig_unités)
                    st.write("**Top catégories avec le plus d'unités:**")
                    st.dataframe(stats_unités.head(5))
                except Exception as e:
                    st.error(f"Erreur: {e}")

        # B. RÉSULTATS DU MODÈLE ENRICHI (F1 Score, Matrice de Confusion)
        st.subheader(" 2. Résultats du Modèle Baseline")

        results = {
            "kw": {
                "f1_train": 0.493,
                "f1_val": 0.488,
                "delta": 0.000
            },
            "kw + length": {
                "f1_train": 0.524,
                "f1_val": 0.520,
                "delta": 0.032
            },
            "kw + unit": {
                "f1_train": 0.533,
                "f1_val": 0.532,
                "delta": 0.043
            },
            "kw + unit + length": {
                "f1_train": 0.552,
                "f1_val": 0.551,
                "delta": 0.063
            }
        }

        baseline_f1 = results["kw"]["f1_val"]
        best_model = "kw + unit + length"

        f1_val = results[best_model]["f1_val"]
        f1_train = results[best_model]["f1_train"]
        delta = results[best_model]["delta"]

        # ---- Metrics ----
        st.metric(
            "F1 Score (Validation)",
            f"{f1_val:.3f}",
            f"+{delta:.3f}"
        )

        st.markdown("**Comparaison des modèles**")

        comparison_df = pd.DataFrame([
            {
                "Modèle": name,
                "F1 Train": v["f1_train"],
                "F1 Validation": v["f1_val"],
                "Δ vs Baseline": v["delta"],
            }
            for name, v in results.items()
        ])

        comparison_df = comparison_df.sort_values("F1 Validation", ascending=False)

        st.dataframe(
            comparison_df,
            use_container_width=True
        )


        st.success(
            f"🏆 Meilleur modèle : **{best_model}** "
            f"(F1 validation = {f1_val:.3f}, +{delta:.3f} vs baseline)"
        )

        # # Options pour l'évaluation
        # col_size, col_btn = st.columns([2, 1])
        # with col_size:
        #     sample_size = st.slider("Taille échantillon", 1000, 5000, 1000, 500, key="full_model_sample")
        # with col_btn:
        #     run_eval = st.button("Lancer l'évaluation du modèle enrichi", key="eval_full_model", type="primary")
        
        # if run_eval:
        #     with st.spinner("Évaluation en cours..."):
        #         try:
        #             # Utiliser ta fonction existante d'enrichissement
        #             results = evaluate_feature_enrichment(df, sample_size=sample_size)
                    
        #             if results and 'kw + unités + longueur' in results:
        #                 # 1. Afficher les scores F1
        #                 f1_train = results['kw + unités + longueur']['f1_train']
        #                 f1_val = results['kw + unités + longueur']['f1_val']
        #                 baseline_f1 = results.get('kw seul', {}).get('f1_val', 0)
                        
        #                 col1, col2, col3 = st.columns(3)
        #                 col1.metric("F1 Score (Validation)", f"{f1_val:.3f}")
        #                 col2.metric("F1 Score (Train)", f"{f1_train:.3f}")
        #                 col3.metric("Amélioration vs Baseline", 
        #                           f"{(f1_val - baseline_f1):.3f}",
        #                           f"{(f1_val - baseline_f1)/baseline_f1*100:.1f}%" if baseline_f1 > 0 else "N/A")
                        
        #                 # 2. Tableau synthèse des performances
        #                 st.markdown("**Comparaison des modèles:**")
        #                 comparison_df = pd.DataFrame({
        #                     'Modèle': ['Baseline (Keywords seul)', 'Modèle Enrichi (Complet)'],
        #                     'F1 Train': [
        #                         results.get('kw seul', {}).get('f1_train', 0),
        #                         f1_train
        #                     ],
        #                     'F1 Validation': [
        #                         results.get('kw seul', {}).get('f1_val', 0),
        #                         f1_val
        #                     ],
        #                     'Nombre Features': [
        #                         results.get('kw seul', {}).get('n_features', 'N/A'),
        #                         results['kw + unités + longueur'].get('n_features', 'N/A')
        #                     ]
        #                 })
        #                 st.dataframe(comparison_df)
                        
        #                 # 3. Graphique d'amélioration
        #                 st.markdown("**Amélioration relative par combinaison:**")
        #                 fig_delta = plot_improvement_delta(results)
        #                 if fig_delta:
        #                     st.pyplot(fig_delta)
                        
                        
        #             else:
        #                 st.warning("Résultats non disponibles pour le modèle complet")
                        
        #         except Exception as e:
        #             st.error(f"Erreur lors de l'évaluation: {str(e)}")

    with tabs[1]:
        st.subheader("Features image — Couleurs")

        # On suppose exactement 2 catégories dans le dictionnaire
        categories = list(hist_dict.keys())


        # ========= LIGNE 1 =========
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Histogramme à dominante bleue**  \n"
                        "_Les images contiennent majoritairement des tons bleus, "
                        "représentatifs de l’eau._")
            fig = plot_rgb_histogram(
                hist_dict["Piscine"],
                title="Piscine",
                y_max=350
            )
            st.pyplot(fig)

        with col2:
            st.markdown("**Histogramme à dominante sombre**  \n"
                        "_Présence marquée de couleurs foncées, typiques des interfaces "
                        "et visuels de jeux vidéo._")
            fig = plot_rgb_histogram(
                hist_dict["Jeux PC"],
                title="Jeux PC",
                y_max=350
            )
            st.pyplot(fig)

        # Espace vertical entre les deux lignes
        st.markdown("<br><br><br>", unsafe_allow_html=True)



        # ========= LIGNE 2 =========
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Histogramme lissé — répartition homogène des couleurs**  \n"
                        "_Les intensités RGB sont réparties de manière continue._")
            fig = plot_rgb_histogram(
                hist_dict["Journaux & magazines"],
                title="Journaux & magazines",
                y_max=350
            )
            st.pyplot(fig)

        with col2:
            st.markdown("**Histogramme avec pics marqués — répartition non uniforme des couleurs**  \n"
                        "_Certaines intensités dominent fortement l’image._")
            fig = plot_rgb_histogram(
                hist_dict["Livres techniques"],
                title="Livres techniques",
                y_max=350
            )
            st.pyplot(fig)

        st.markdown("### Exemple de l'image d'un objet monochrome — Livres spécialisés")

        st.markdown(
            "Cet exemple illustre l’origine des **pics observés dans les histogrammes RGB** "
            "au niveau de la catégorie *Livres spécialisés*."
        )

        col_img, col_hist = st.columns([1, 1])

        with col_img:
            left, center, right = st.columns([1, 2, 1])
            with center:
                st.markdown("<br><br><br>", unsafe_allow_html=True)
                display_image(
                    example["image"],
                    caption="Image d’un manuscrit",
                    width=400
                )

        with col_hist:
            fig = plot_rgb_histogram(
                example["hist_rgb"],
                title="Histogramme RGB de l’image"
            )
            st.pyplot(fig)


        with tabs[2]:
            st.subheader("Features image — Formes et intensité")

            st.markdown(
                "Comparaison des **images moyennes** pour des catégories visuellement contrastées."
            )

            # ------------------------------------------------------------
            # Affichage des images moyennes côte à côte
            # ------------------------------------------------------------

            categories_to_show = {   
                'Jeux PC': 0.792,
                'Cartes à collectionner': 0.778,
                'Déco & Éclairage': 0.115
            }

            cols = st.columns(len(categories_to_show))

            for col, category in zip(cols, categories_to_show):
                with col:
                    st.markdown(f"**{category}**")
                    display_image(mean_intensity[category], caption='intensité moyenne')

            st.markdown("#### Taux de formes rectangulaires détectées")

            cols = st.columns(3)

            for col, (category, rate) in zip(cols, categories_to_show.items()):
                with col:
                    st.metric(
                        label=category,
                        value=f"{rate:.1%}"
                    )

            # ------------------------------------------------------------
            # Interprétation
            # ------------------------------------------------------------
            st.info(
                "Les jeux PC présentent des images moyennes globalement plus sombres et structurées, "
                "souvent associées à des formes rectangulaires régulières (jaquettes, boîtiers). "
                "À l’inverse, les catégories décoration et luminaires montrent des structures plus "
                "diffuses et hétérogènes. Ces différences motivent l’extraction de features liées "
                "aux formes et aux contours."
            )


        # ============================================================
        # TAB 4 — MODÈLE BASELINE IMAGE
        # ============================================================
        with tabs[3]:

            st.subheader("Modèle baseline image — Résultats et interprétation")

            st.markdown(
                """
                Cette section présente les performances du **modèle image seul**,
                basé sur des descripteurs classiques (couleurs, formes, textures).
                L’objectif est d’évaluer dans quelle mesure l’information visuelle
                est discriminante sans recours au deep learning.
                """
            )

            st.markdown("---")

            # ------------------------------------------------------------
            # Score global
            # ------------------------------------------------------------
            st.markdown("### Performance globale")

            st.metric(
                label="F1-score pondéré — Modèle image seul",
                value="37.3 %"
            )

            st.markdown(
                """
                Le score global reste limité, ce qui indique que les images seules
                ne suffisent pas à discriminer l’ensemble des catégories.
                """
            )

            st.markdown("---")

            # ------------------------------------------------------------
            # Meilleures / pires catégories
            # ------------------------------------------------------------
            st.markdown("### Analyse par catégorie")

            col_good, col_bad = st.columns(2)

            with col_good:
                st.markdown("**Catégories les plus performantes**")
                st.markdown(
                    """
                    - **Cartes à collectionner** : **73.6 %**
                    - **Textiles** : **66.9 %**
                    - **Romans & littérature** : **59.9 %**
                    """
                )

            with col_bad:
                st.markdown("**Catégories les plus difficiles**")
                st.markdown(
                    """
                    - **Jeux éducatifs** : **9 %**  
                    - **Animaux** : **8.8 %**  
                    - **Jeux de rôle** : **8.6 %**  
                    """
                )


            st.info(
                "Les bonnes performances sont observées pour des catégories visuellement "
                "homogènes, tandis que les catégories hétérogènes restent difficiles à "
                "discriminer avec des features visuelles simples."
            )

            st.markdown("---")

            # ------------------------------------------------------------
            # Feature importance — meilleure catégorie uniquement
            # ------------------------------------------------------------
            st.markdown("### Interprétation — Cartes à collectionner")

            col_left, col_center, col_right = st.columns([1, 4, 1])

            with col_center:
                df_norm = df_importance.div(
                    df_importance.sum(axis=1),
                    axis=0
                )

            st.pyplot(plot_feature_importance_heatmap(df_norm))

        
    
# ==================================================================================================================================
# SECTION 6: STRATÉGIE DE MODÉLISATION:
# ==================================================================================================================================
elif choice == "6. Stratégie de modélisation":
    st.header("6. Stratégie de modélisation")

    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 30px;">
    Cette section illustre notre pipeline complet de modélisation multimodale, 
    qui combine intelligemment les modèles de texte et d'image pour la classification des produits.
    </div>
    """, unsafe_allow_html=True)

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    def draw_modeling_strategy():
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Désactiver les axes
        ax.axis('off')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        
        # Définir les couleurs
        colors = {
            'data': '#4CAF50',
            'text': '#2196F3', 
            'image': '#9C27B0',
            'model': '#607D8B',
            'blending': '#FF9800',
            'stacking': '#F44336',
            'output': '#4CAF50'
        }
        
        # Fonction pour dessiner une boîte avec bord arrondi
        def draw_box(x, y, width, height, text, color, fontsize=10):
            box = FancyBboxPatch(
                (x, y), width, height,
                boxstyle="round,pad=0.1,rounding_size=0.05",
                linewidth=1.5,
                edgecolor='#333',
                facecolor=color,
                alpha=0.9
            )
            ax.add_patch(box)
            
            lines = text.split('\n')
            for i, line in enumerate(lines):
                y_pos = y + height/2 + (len(lines)/2 - i - 0.5) * 0.15
                ax.text(x + width/2, y_pos, line,
                       ha='center', va='center',
                       fontsize=fontsize, 
                       fontweight='bold' if i == 0 else 'normal',
                       color='white' if color in ['#4CAF50', '#2196F3', '#9C27B0', '#F44336', '#FF9800'] else 'black')
        
        # Fleches courtes
        def draw_very_short_arrow(x1, y1, x2, y2, color='#333', width=1.5, shorten=0.6):
            
            dx = x2 - x1
            dy = y2 - y1
            length = (dx**2 + dy**2)**0.5
            
         
            start_shorten = shorten * 0.3  
            end_shorten = shorten * 0.7    
            
            x1_short = x1 + dx * start_shorten
            y1_short = y1 + dy * start_shorten
            x2_short = x2 - dx * end_shorten
            y2_short = y2 - dy * end_shorten
            
            arrow = FancyArrowPatch(
                (x1_short, y1_short), (x2_short, y2_short),
                arrowstyle='->',
                color=color,
                linewidth=width,
                mutation_scale=10  
            )
            ax.add_patch(arrow)
        
        # =========================================
        # TÍTRE
        # =========================================
        ax.text(5, 11.5, "PIPELINE DE MODÉLISATION MULTIMODALE",
               ha='center', va='center', fontsize=16, fontweight='bold', color='#003366')
        
        ax.text(5, 11.1, "",
               ha='center', va='center', fontsize=12, color='#666', style='italic')
        
        ax.plot([3, 7], [10.8, 10.8], color='#d9822b', linewidth=2, alpha=0.7)
        

        # NIVEAU 1: DONNÉES RENSEIGNÉES
        draw_box(4.25, 9.3, 1.5, 0.6, "Données\nrenseignées", colors['data'], fontsize=11)
        
        # =========================================
        # NIVEAU 2: TEXTE ET IMAGE 
        # =========================================
        # Texte
        draw_box(2, 8.2, 1.2, 0.5, "Texte", colors['text'], fontsize=10)
        
        # Image  
        draw_box(6.5, 8.2, 1.2, 0.5, "Image", colors['image'], fontsize=10)
        
        # =========================================
        # NIVEAU 3: MODÈLES 
        # =========================================
        # Modèles texte
        draw_box(1, 7.0, 1, 0.5, "TF-IDF\nSVM", colors['model'], fontsize=9)
        draw_box(2.5, 7.0, 1, 0.5, "Camembert", colors['model'], fontsize=9)
        draw_box(4, 7.0, 1, 0.5, "XLM-Roberta", colors['model'], fontsize=9)
        
        # Modèles image
        draw_box(6, 7.0, 1, 0.5, "Swin", colors['model'], fontsize=9)
        draw_box(7.5, 7.0, 1, 0.5, "ConvNext", colors['model'], fontsize=9)
        
        # =========================================
        # NIVEAU 4: BLENDING 
        # =========================================
        # Blending texte
        draw_box(2.5, 5.8, 1.2, 0.5, "Blending", colors['blending'], fontsize=10)
        
        # Blending image
        draw_box(6.5, 5.8, 1.2, 0.5, "Blending", colors['blending'], fontsize=10)
        
        # =========================================
        # NIVEAU 5: STACKING 
        # =========================================
        draw_box(4, 4.6, 2, 0.6, "Stacking\nLogistic regression", colors['stacking'], fontsize=10)
        
        # =========================================
        # NIVEAU 6: SORTIE 
        # =========================================
        draw_box(4, 3.4, 2, 0.6, "Estimation\ncode catégorie\ndu produit", colors['output'], fontsize=10)
        
        # =========================================
        # FLÈCHES COURTES
        # =========================================
        # Données → Texte/Image 
        draw_very_short_arrow(5, 9.3, 2.6, 8.45)
        draw_very_short_arrow(5, 9.3, 7.1, 8.45)
        
        # Texte → Modèles texte 
        draw_very_short_arrow(2.6, 8.2, 1.5, 7.25)
        draw_very_short_arrow(2.6, 8.2, 3, 7.25)
        draw_very_short_arrow(2.6, 8.2, 4.5, 7.25)
        
        # Image → Modèles image 
        draw_very_short_arrow(7.1, 8.2, 6.5, 7.25)
        draw_very_short_arrow(7.1, 8.2, 8, 7.25)
        
        # Modèles texte → Blending texte
        draw_very_short_arrow(1.5, 7.0, 3.1, 6.05)
        draw_very_short_arrow(3, 7.0, 3.1, 6.05)
        draw_very_short_arrow(4.5, 7.0, 3.1, 6.05)
        
        # Modèles image → Blending image 
        draw_very_short_arrow(6.5, 7.0, 7.1, 6.05)
        draw_very_short_arrow(8, 7.0, 7.1, 6.05)
        
        # Blending texte → Stacking 
        draw_very_short_arrow(3.1, 5.8, 4.5, 4.9)
        
        # Blending image → Stacking 
        draw_very_short_arrow(7.1, 5.8, 5.5, 4.9)
        
        # Stacking → Estimation 
        draw_very_short_arrow(5, 4.6, 5, 4.0)
        
        # =========================================
        # LÉGENDE
        # =========================================
        legend_y = 2.0
        legend_spacing = 1.4
        
        ax.text(5, 2.7, "Légende des couleurs",
               ha='center', va='center', fontsize=10, fontweight='bold', color='#003366')
        
        legend_items = [
            ("Données", colors['data']),
            ("Texte", colors['text']),
            ("Image", colors['image']),
            ("Modèles", colors['model']),
            ("Fusion", colors['blending']),
            ("Stacking", colors['stacking'])
        ]
        
        for i, (label, color) in enumerate(legend_items):
            ax.add_patch(plt.Rectangle((i*legend_spacing + 0.5, legend_y), 0.25, 0.25, 
                                     facecolor=color, edgecolor='#333', alpha=0.9))
            ax.text(i*legend_spacing + 0.85, legend_y + 0.125, label,
                   ha='left', va='center', fontsize=9)
        
        ax.text(5, 1.6, "Chaque modalité est traitée séparément puis fusionnée pour une prédiction optimale",
               ha='center', va='center', fontsize=9, color='#666', style='italic')
        
        plt.tight_layout(pad=3.0)
        return fig

    # Afficher le graphique
    fig = draw_modeling_strategy()
    st.pyplot(fig, use_container_width=True)
    

# ==================================================================================================================================
# SECTION 7: VECTORISATION & CHOIX DE FEATURES 
# ==================================================================================================================================


elif choice == "7. Vectorisation ":
    st.header("7. Vectorisation & choix de features")

    # --- Imports locaux ---
    import re as _re
    import numpy as _np
    import pandas as _pd
    import plotly.express as _px

    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import f1_score, classification_report
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    from sklearn.pipeline import FeatureUnion

    # --- Composants interactifs (optionnels) ---
    _HAS_AGGRID = False
    _HAS_ACE = False
    _HAS_AGRAPH = False
    try:
        from st_aggrid import AgGrid  # type: ignore
        _HAS_AGGRID = True
    except Exception:
        pass
    try:
        from streamlit_ace import st_ace  # type: ignore
        _HAS_ACE = True
    except Exception:
        pass
    try:
        from streamlit_agraph import agraph, Node, Edge, Config  # type: ignore
        _HAS_AGRAPH = True
    except Exception:
        pass

    # --- langdetect (FR / Multilingue) ---
    try:
        from langdetect import detect as _langdetect_detect  # type: ignore
        _LANGDETECT_OK = True
    except Exception:
        _LANGDETECT_OK = False

    def _bucket_lang_fr_mult(txt: str) -> str:
        if not isinstance(txt, str) or not txt.strip():
            return "Multilingue"
        if not _LANGDETECT_OK:
            t = txt.lower()
            if any(ch in t for ch in "éàèùêâîôç") or " le " in t or " la " in t:
                return "FR"
            return "Multilingue"
        try:
            return "FR" if _langdetect_detect(txt) == "fr" else "Multilingue"
        except Exception:
            return "Multilingue"

    _UNIT_RE = _re.compile(r"(?i)\b(cm|mm|m|kg|g|mg|l|ml|cl|gb|tb|hz|w|kw|v|mah|mp|px|inch|in)\b")

    def _has_digits(txt: str) -> bool:
        return bool(_re.search(r"\d", txt or ""))

    def _has_units(txt: str) -> bool:
        return bool(_UNIT_RE.search(txt or ""))

    def _infer_cat_cols(_df: _pd.DataFrame):
        cols = {c.lower(): c for c in _df.columns}
        group_candidates = ["groupe", "group", "supercat", "super_cat", "categorie_parent", "category_group"]
        cat_candidates   = ["categorie", "category", "sous_categorie", "subcategory", "sous-cat", "sub_category"]
        group_col = next((cols[c] for c in group_candidates if c in cols), None)
        cat_col   = next((cols[c] for c in cat_candidates if c in cols), None)
        return group_col, cat_col

    def _code_to_name_map(_df: _pd.DataFrame) -> dict:
        group_col, cat_col = _infer_cat_cols(_df)
        if group_col and cat_col:
            tmp = _df[["prdtypecode", group_col, cat_col]].dropna().copy()
            def _mode(s):
                m = s.mode()
                return m.iloc[0] if not m.empty else s.iloc[0]
            agg = tmp.groupby("prdtypecode").agg({group_col:_mode, cat_col:_mode})
            return {int(k): f"{row[group_col]} ▸ {row[cat_col]}" for k,row in agg.iterrows()}
        return {int(k): str(int(k)) for k in _df["prdtypecode"].unique()}

    _CODE2NAME = _code_to_name_map(df)

    def _clean_spans(s: str) -> str:
        if not isinstance(s, str):
            return s
        # remove html tags like <span ...>
        s = _re.sub(r"<[^>]+>", "", s)
        return s.strip()

    def _make_text(series_a, series_b):
        a = series_a.fillna("").astype(str)
        b = series_b.fillna("").astype(str)
        return (a + " " + b).str.strip()

    # ------------------------------------------------------------------
    # Header (style pièce jointe)
    # ------------------------------------------------------------------
    st.markdown(f"""
    <div style=\"background: linear-gradient(135deg, rgba(0,51,102,0.08), rgba(217,130,43,0.10));
                padding: 18px 18px; border-radius: 14px; border: 1px solid rgba(0,0,0,0.06);\">
      <h2 style=\"margin:0;color:{COLORS['primary']};\">Vectorisation texte : du brut au signal utile</h2>
      <p style=\"margin:6px 0 0 0; color:#334; font-size: 0.98em;\">
        On ne choisit pas seulement un modèle : on choisit un <b>pipeline</b> (représentation → classifieur → calibration).
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Pipeline Build vs Evaluate (Graphviz)
    # ------------------------------------------------------------------
    st.subheader("Pipeline : Build vs Evaluate")
    dot = r"""
    digraph G {
      rankdir=LR;
      node  [shape=box, style="rounded", fontname="Arial", fontsize=10, color="#003366"];
      edge  [color="#333333", arrowsize=0.8];

      subgraph cluster_build {
        label="BUILD (construire le signal)";
        color="#d9822b"; style="rounded"; fontsize=11; fontcolor="#003366";

        data  [label="Données texte\n(titre + description)", style="rounded,filled", fillcolor="#ffffff"];
        split [label="Split titre / description", style="rounded,filled", fillcolor="#ffffff"];
        char  [label="Char n-grams", style="rounded,filled", fillcolor="#ffffff"];
        word  [label="TF-IDF (word)", style="rounded,filled", fillcolor="#ffffff"];
        model [label="LinearSVC", style="rounded,filled", fillcolor="#003366", fontcolor="#ffffff"];

        data -> split;
        split -> char;
        split -> word;
        char -> model;
        word -> model;
      }

      subgraph cluster_eval {
        label="EVALUATE (prouver & comprendre)";
        color="#d9822b"; style="rounded"; fontsize=11; fontcolor="#003366";

        err  [label="Erreurs & Explications\n(top mots)", style="rounded,filled", fillcolor="#ffffff"];
        cal  [label="Calibration\n(probas fiables)", style="rounded,filled", fillcolor="#ffffff"];
        time [label="Temps train / prédiction\n(coût)", style="rounded,filled", fillcolor="#ffffff"];
        f1   [label="F1 weighted\n(métrique principale)", style="rounded,filled", fillcolor="#ffffff"];
      }

      model -> err;
      model -> cal;
      model -> time;
      model -> f1;
    }
    """
    st.graphviz_chart(dot, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Baseline tableau (trié F1 décroissant)
    # ------------------------------------------------------------------

    st.subheader("Baseline linéaire : performance vs coût")

    baseline_models = _pd.DataFrame([
        {"model":"LinearSVC","f1_train":0.991,"f1_val":0.838,"train_time_s":28.044,"pred_time_s":11.093},
        {"model":"RidgeClassifier","f1_train":0.977,"f1_val":0.833,"train_time_s":73.428,"pred_time_s":10.825},
        {"model":"LogisticRegression","f1_train":0.903,"f1_val":0.816,"train_time_s":234.674,"pred_time_s":17.124},
        {"model":"SGD_hinge","f1_train":0.893,"f1_val":0.803,"train_time_s":18.731,"pred_time_s":14.154},
        {"model":"ComplementNB","f1_train":0.866,"f1_val":0.772,"train_time_s":13.735,"pred_time_s":11.433},
    ]).sort_values("f1_val", ascending=False)

    colL, colR = st.columns([3,1])
    with colL:
        fig = _px.scatter(
            baseline_models,
            x="train_time_s",
            y="f1_val",
            color="model",
            size="pred_time_s",
            hover_data=["f1_train", "train_time_s", "pred_time_s"],
            title="Performance vs Coût des modèles baseline",
        )
        fig.update_layout(height=420, xaxis_title="train_time_s", yaxis_title="f1_val")
        st.plotly_chart(fig, use_container_width=True)

    with colR:
        st.markdown("""
        <div style="background:#eaf7ee;border:1px solid rgba(0,0,0,0.08);padding:14px;border-radius:14px;">
          <h4 style="margin:0;color:#1b5e20;">✅ Choix retenu : LinearSVC</h4>
          <ul style="margin:8px 0 0 18px;color:#2b2b2b;">
            <li><b>Meilleur F1 validation</b> (0.838)</li>
            <li><b>Coût raisonnable</b> (train 28.0s)</li>
            <li>Baseline idéale</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    table_df = baseline_models.rename(columns={
        "model":"Modèles",
        "f1_train":"F1 train",
        "f1_val":"F1 validation",
        "train_time_s":"Temps entraînement (s)",
        "pred_time_s":"Temps prédiction (s)",
    })
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Split titre/description (calculé sur un sous-échantillon)
    # ------------------------------------------------------------------
    st.subheader("Split titre/description : quel champ porte le signal ?")
    split_note = """
    <div style="background:#eaf7ee;border:1px solid rgba(0,0,0,0.08);padding:14px;border-radius:14px;">
      <h4 style="margin:0;color:#1b5e20;">✅ Choix retenu : titre + description</h4>
      <p style="margin:6px 0 0 0;color:#2b2b2b;">
        La fusion maximise le rappel utile : le titre est dense, la description apporte du contexte.
      </p>
    </div>
    """

    sample_n = st.slider("Taille d'échantillon (split ablation)", 5000, min(40000, len(df)), 15000, 5000)
    do_split_eval = st.button("Calculer l'ablation split", key="split_eval_btn")

    @st.cache_data(show_spinner=False)
    def _split_ablation(_n):
        title_col = "designation" if "designation" in df.columns else next((c for c in df.columns if "design" in c.lower()), None)
        desc_col  = "description" if "description" in df.columns else next((c for c in df.columns if "descr" in c.lower()), None)
        if title_col is None or desc_col is None:
            return _pd.DataFrame([])

        sub = df[[title_col, desc_col, "prdtypecode"]].dropna(subset=["prdtypecode"]).copy()
        if _n < len(sub):
            sub = sub.sample(n=_n, random_state=42)

        y = sub["prdtypecode"].astype(int).values

        def _eval(text_series):
            X = text_series.fillna("").astype(str).tolist()
            Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            vect = TfidfVectorizer(ngram_range=(1,2), min_df=2)
            clf = LinearSVC(C=0.2)
            pipe = Pipeline([("vect", vect), ("clf", clf)])
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xva)
            return float(f1_score(yva, pred, average="weighted"))

        f_title = _eval(sub[title_col])
        f_desc  = _eval(sub[desc_col])
        f_both  = _eval(_make_text(sub[title_col], sub[desc_col]))

        return _pd.DataFrame([
            {"champ":"titre", "f1_weighted":f_title},
            {"champ":"description", "f1_weighted":f_desc},
            {"champ":"titre + description", "f1_weighted":f_both},
        ]).sort_values("f1_weighted", ascending=False)

    left,right = st.columns([3,1])
    with left:
        if do_split_eval:
            res = _split_ablation(sample_n)
            if res.empty:
                st.warning("Colonnes texte introuvables pour calculer l'ablation split.")
            else:
                st.plotly_chart(_px.bar(res, x="f1_weighted", y="champ", orientation="h", title="F1 weighted (val)"),
                                use_container_width=True)
        # else:
        #     st.info("Calculer pour afficher le graphique.")
    with right:
        st.markdown(split_note, unsafe_allow_html=True)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Ablation vectorizer
    # ------------------------------------------------------------------
    st.subheader("Ablation : choix du vectorizer")
    vect_df = _pd.DataFrame([
        {"vectorizer":"tfidf","f1_val":0.838,"lecture":"signal stable"},
        {"vectorizer":"count_binary","f1_val":0.818,"lecture":"présence/absence"},
        {"vectorizer":"count","f1_val":0.808,"lecture":"comptage brut"},
    ]).sort_values("f1_val", ascending=False)

    c1,c2 = st.columns([3,1])
    with c1:
        st.dataframe(vect_df, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("""
        <div style="background:#eaf7ee;border:1px solid rgba(0,0,0,0.08);padding:14px;border-radius:14px;">
          <h4 style="margin:0;color:#1b5e20;">✅ Décision : TF-IDF</h4>
          <p style="margin:6px 0 0 0;color:#2b2b2b;">
            Meilleur F1 weighted / F1_val parmi les options testées : compromis simple et défendable.
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ------------------------------------------------------------------
    # N-grams
    # ------------------------------------------------------------------
    st.subheader("N-grams ")
    ngram_df = _pd.DataFrame([
        {"config":"word_1_2gram","f1_val":0.838},
        {"config":"word_1_3gram","f1_val":0.837},
        {"config":"word_1gram","f1_val":0.833},
    ]).sort_values("f1_val", ascending=False)

    c1,c2 = st.columns([3,1])
    with c1:
        cols = st.columns(3)
        for i,row in enumerate(ngram_df.itertuples(index=False)):
            with cols[i]:
                st.markdown(f"<div style='font-size:0.9em;color:#555;'><b>{row.config}</b></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:2.4em;color:#111;margin-top:-4px;'>{row.f1_val:.3f}</div>", unsafe_allow_html=True)
                st.progress(min(max((row.f1_val-0.80)/0.08, 0), 1.0))
    with c2:
        st.markdown("""
        <div style="background:#eaf7ee;border:1px solid rgba(0,0,0,0.08);padding:14px;border-radius:14px;">
          <h4 style="margin:0;color:#1b5e20;">✅ Décision : word n-grams (1–2)</h4>
          <p style="margin:6px 0 0 0;color:#2b2b2b;">
            Le gain 1–3 est marginal : on retient la configuration la plus simple (moins de bruit / coût).
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Word + Char n-grams
    # ------------------------------------------------------------------
    st.subheader("Word + Char n-grams ")
    c1,c2 = st.columns([3,1])
    with c1:
        a,b = st.columns(2)
        with a:
            st.markdown("**Avant**")
            st.markdown("<div style='color:#555;'>word_only</div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:2.4em;color:#111;'>0.838</div>", unsafe_allow_html=True)
        with b:
            st.markdown("**Après**")
            st.markdown("<div style='color:#555;'>word + char(3–5)</div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:2.4em;color:#111;'>0.860</div>", unsafe_allow_html=True)
            st.markdown("<span style='background:#eaf7ee;padding:4px 10px;border-radius:999px;color:#1b5e20;border:1px solid rgba(0,0,0,0.08);'>↑ +0.022</span>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#f8f9fa;border:1px solid rgba(0,0,0,0.06);padding:14px;border-radius:14px;margin-top:14px;">
          <h4 style="margin:0;color:#003366;">Pourquoi ça marche ?</h4>
          <p style="margin:6px 0 0 0;color:#333;">
            Les <b>char n-grams</b> capturent les références, les fautes, les variantes, les tailles, les versions, les codes produit robustes au bruit et utiles sur titres courts.
          </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div style="background:#eaf7ee;border:1px solid rgba(0,0,0,0.08);padding:14px;border-radius:14px;">
          <h4 style="margin:0;color:#1b5e20;">✅ Décision : word + char(3–5)</h4>
          <p style="margin:6px 0 0 0;color:#2b2b2b;">
            Gain net F1 : meilleure robustesse (typos, codes, tailles). Défendable en termes de signal capturé.
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ------------------------------------------------------------------
    # C : contrôler la complexité
    # ------------------------------------------------------------------
    st.subheader("C : contrôler la complexité sans perdre en généralisation")
    C_GRID = [(0.01, 0.8349), (0.05, 0.8359), (0.1, 0.8416), (0.2, 0.8418), (0.3, 0.8400), (0.5, 0.8384)]
    c_df = _pd.DataFrame({"C":[c for c,_ in C_GRID], "f1_val":[f for _,f in C_GRID]})

    left,right = st.columns([3,1])
    with left:
        fig = _px.line(c_df, x="C", y="f1_val", markers=True, title="Grid search (LinearSVC) : F1 vs C")
        fig.update_layout(yaxis_title="F1 validation", xaxis_title="C", height=360)
        st.plotly_chart(fig, use_container_width=True)
        # st.caption("Astuce : le coût (temps train/pred) augmente généralement avec C. Ici, compromis autour de C≈0.2.")
    with right:
        st.markdown("""
        <div style="background:#eaf7ee;border:1px solid rgba(0,0,0,0.08);padding:14px;border-radius:14px;">
          <h4 style="margin:0;color:#1b5e20;">Décision : C = 0.2</h4>
          <ul style="margin:8px 0 0 18px;color:#2b2b2b;">
            <li>C trop faible → sous-apprentissage (frontière trop lisse).</li>
            <li>C trop élevé → risque d’overfit + coût plus élevé.</li>
            <li>On privilégie le <b>compromis perf/coût</b>.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Calibration + KPIs features (avec espace)
    # ------------------------------------------------------------------
    st.subheader("Avant / Après calibration : fiabilité des scores")
    cal_df = _pd.DataFrame([
        {"metric":"F1 weighted","avant":0.8696,"après":0.8674},
        {"metric":"LogLoss","avant":1.679,"après":0.501},
    ])

    l,r = st.columns([3,1])
    with l:
        fig = _px.bar(cal_df.melt(id_vars=["metric"], var_name="stage", value_name="value"),
                     x="metric", y="value", color="stage", barmode="group",
                     title="Avant / Après calibration")
        fig.update_layout(height=330, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        # st.markdown("""
        # <div style="background:#f8f9fa;border:1px solid rgba(0,0,0,0.06);padding:14px;border-radius:14px;">
        #   <h4 style="margin:0;color:#003366;">Interprétation</h4>
        #   <p style="margin:6px 0 0 0;color:#333;">
        #     Le <b>F1 weighted</b> bouge peu, mais la <b>LogLoss chute</b> : scores plus fiables (utile pour routage, seuils, fusion).
        #   </p>
        # </div>
        # """, unsafe_allow_html=True)

        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

        k1,k2,k3 = st.columns(3)
        with k1:
            st.markdown("<div style='background:white;border:1px solid rgba(0,0,0,0.06);padding:14px;border-radius:14px;'>"
                        "<div style='color:#666;font-size:0.9em;'>Baseline texte (NB01)</div>"
                        "<div style='font-size:2.0em;color:#003366;'><b>0.792</b></div>"
                        "<div style='color:#666;font-size:0.85em;'>référence challenge</div></div>", unsafe_allow_html=True)
        with k2:
            st.markdown("<div style='background:white;border:1px solid rgba(0,0,0,0.06);padding:14px;border-radius:14px;'>"
                        "<div style='color:#666;font-size:0.9em;'>Meilleur pipeline de vectorisation</div>"
                        "<div style='font-size:2.0em;color:#003366;'><b>0.844</b></div>"
                        "<div style='color:#666;font-size:0.85em;'>TF-IDF + poids titre + features</div></div>", unsafe_allow_html=True)
        with k3:
            st.markdown("<div style='background:white;border:1px solid rgba(0,0,0,0.06);padding:14px;border-radius:14px;'>"
                        "<div style='color:#666;font-size:0.9em;'>Gain global</div>"
                        "<div style='font-size:2.0em;color:#003366;'><b>+6.5%</b></div>"
                        "<div style='color:#666;font-size:0.85em;'>de 0.792 à 0.844</div></div>", unsafe_allow_html=True)

    with r:
        st.markdown("""
        <div style="background:#eaf7ee;border:1px solid rgba(0,0,0,0.08);padding:14px;border-radius:14px;">
          <h4 style="margin:0;color:#1b5e20;">Pourquoi F1 weighted ?</h4>
          <p style="margin:6px 0 0 0;color:#2b2b2b;">
            Classes déséquilibrées : F1 weighted pondère par support → c'est la métrique la plus représentative en production.
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Studio explicabilité : slices + top-K contributions (FIX coo_matrix)
    # ------------------------------------------------------------------
    st.subheader("Qu'est-ce qui influence le modèle?")

    s1,s2,s3 = st.columns(3)
    with s1:
        lang_opt = st.selectbox("Slice langue", ["Aucun filtre", "FR", "Multilingue"], index=0)
    with s2:
        dig_opt = st.selectbox("Slice chiffres", ["Aucun filtre", "Avec chiffres", "Sans chiffres"], index=0)
    with s3:
        unit_opt = st.selectbox("Slice unités", ["Aucun filtre", "Avec unités", "Sans unités"], index=0)

    active = []
    if lang_opt != "Aucun filtre": active.append(lang_opt)
    if dig_opt != "Aucun filtre": active.append(dig_opt)
    if unit_opt != "Aucun filtre": active.append(unit_opt)
    st.markdown(f"**Sous-ensemble analysé :** {' • '.join(active) if active else 'Aucun filtre'}")

    audit_n = st.slider("Taille d'audit", 5000, min(80000, len(df)), min(20000, len(df)), 5000)
    topk = st.slider("Top-K contributions affichées", 5, 30, 12, 1)
    run = st.button("Analyser", type="primary")

    @st.cache_resource(show_spinner=False)
    def _train_audit(_n, _lang, _dig, _unit):
        title_col = "designation" if "designation" in df.columns else next((c for c in df.columns if "design" in c.lower()), df.columns[0])
        desc_col  = "description" if "description" in df.columns else next((c for c in df.columns if "descr" in c.lower()), title_col)

        _df = df[[title_col, desc_col, "prdtypecode"]].copy()
        _df["text"] = _make_text(_df[title_col], _df[desc_col])

        if _lang != "Aucun filtre":
            _df["lang_bucket"] = _df["text"].map(_bucket_lang_fr_mult)
            _df = _df[_df["lang_bucket"] == _lang]
        if _dig != "Aucun filtre":
            _df["has_dig"] = _df["text"].map(_has_digits)
            _df = _df[_df["has_dig"] == ("Avec chiffres" == _dig)]
        if _unit != "Aucun filtre":
            _df["has_unit"] = _df["text"].map(_has_units)
            _df = _df[_df["has_unit"] == ("Avec unités" == _unit)]

        vc = _df["prdtypecode"].value_counts()
        _df = _df[_df["prdtypecode"].isin(vc[vc >= 2].index)]
        if _n < len(_df):
            _df = _df.sample(n=_n, random_state=42)

        X = _df["text"].tolist()
        y = _df["prdtypecode"].astype(int).tolist()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        word_vect = TfidfVectorizer(ngram_range=(1,2), min_df=2)
        char_vect = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2)
        feats = FeatureUnion([("word", word_vect), ("char", char_vect)])
        clf = LinearSVC(C=0.2)
        pipe = Pipeline([("feats", feats), ("clf", clf)])

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)
        f1w = f1_score(y_val, pred, average="weighted")

        scores = None
        try:
            scores = pipe.decision_function(X_val)
        except Exception:
            pass

        return {"pipe":pipe,"X_val":X_val,"y_val":y_val,"pred":pred,"scores":scores,"f1w":f1w}


    # --- Persistance des résultats d'audit (évite de relancer l'entraînement quand on change une sélection) ---
    _AUDIT_STATE_KEY = 'vec_audit_state'
    _params = (audit_n, lang_opt, dig_opt, unit_opt)
    if _AUDIT_STATE_KEY not in st.session_state:
        st.session_state[_AUDIT_STATE_KEY] = {'params': None, 'bundle': None}

    if run:
        with st.spinner("Entraînement du modèle d'audit…"):
            bundle = _train_audit(audit_n, lang_opt, dig_opt, unit_opt)
        st.session_state[_AUDIT_STATE_KEY] = {'params': _params, 'bundle': bundle}

    _state = st.session_state.get(_AUDIT_STATE_KEY, {})
    bundle = _state.get('bundle') if _state.get('params') == _params else None


    if bundle is not None:
        st.success(f"F1 weighted: {bundle['f1w']:.3f}")

        rep = classification_report(bundle["y_val"], bundle["pred"], output_dict=True, zero_division=0)
        rep_df = _pd.DataFrame(rep).T.drop(index=["accuracy","macro avg","weighted avg"], errors="ignore").reset_index()
        rep_df = rep_df.rename(columns={"index":"prdtypecode","f1-score":"f1"})[["prdtypecode","f1","precision","recall","support"]]
        rep_df["prdtypecode"] = rep_df["prdtypecode"].astype(int)
        rep_df.insert(0, "Catégorie", rep_df["prdtypecode"].map(lambda c: _clean_spans(_CODE2NAME.get(int(c), str(c)))))
        rep_df = rep_df.sort_values("f1", ascending=False)

        st.markdown("### Performance par catégorie")
        st.dataframe(rep_df.head(25), use_container_width=True, hide_index=True)

        st.markdown("### Exemples d'erreurs")
        errs=[]
        for i,(yt,yp) in enumerate(zip(bundle["y_val"], bundle["pred"])):
            if yt != yp:
                conf=0.0
                if bundle["scores"] is not None:
                    row=bundle["scores"][i]
                    best=float(_np.max(row))
                    second=float(_np.partition(row,-2)[-2]) if len(row)>=2 else best
                    conf=best-second
                errs.append({
                    'idx':i,
                    'Vrai':_clean_spans(_CODE2NAME.get(int(yt),str(yt))),
                    'Prédit':_clean_spans(_CODE2NAME.get(int(yp),str(yp))),
                    'marge_confiance':conf,
                    'texte':bundle['X_val'][i][:240]+('…' if len(bundle['X_val'][i])>240 else '')
                })
        err_df=_pd.DataFrame(errs).sort_values('marge_confiance', ascending=False).head(30)
        st.dataframe(err_df, use_container_width=True, hide_index=True)

        if not err_df.empty:
            st.markdown("### Top contributions → classe prédite")
            err_df["choix"] = err_df.apply(lambda r: f"{r['Vrai']} → {r['Prédit']} (marge {r['marge_confiance']:.3f})", axis=1)
            _choice = st.selectbox("Choisir une erreur", err_df["choix"].tolist(), index=0)
            pick = int(err_df.loc[err_df["choix"] == _choice, "idx"].iloc[0])
            txt = bundle["X_val"][int(pick)]

            pipe=bundle["pipe"]
            feats=pipe.named_steps["feats"]
            clf=pipe.named_steps["clf"]

            Xrow = feats.transform([txt])

            pred_label=int(bundle["pred"][int(pick)])
            true_label=int(bundle["y_val"][int(pick)])

            def _class_index(label):
                return int(_np.where(clf.classes_==label)[0][0])

            def _top_word_contrib(label):
                ci=_class_index(label)
                w=clf.coef_[ci]
                contrib = Xrow.multiply(w).tocsr()  # ✅ FIX COO -> CSR

                word_names = feats.transformer_list[0][1].get_feature_names_out()
                n_word=len(word_names)

                inds=contrib.indices
                vals=contrib.data
                items=[]
                for j,v in zip(inds, vals):
                    if j < n_word:
                        items.append((word_names[j], float(v)))
                items.sort(key=lambda x:x[1], reverse=True)
                return items[:topk]

            cA,cB,cC = st.columns(3)
            with cA:
                st.markdown("**Pourquoi le modèle a prédit cette classe?**")
                st.dataframe(_pd.DataFrame(_top_word_contrib(pred_label), columns=["token","contribution"]),
                             use_container_width=True, hide_index=True)
            with cB:
                st.markdown("**Ce qui aurait soutenu la vraie classe:**")
                st.dataframe(_pd.DataFrame(_top_word_contrib(true_label), columns=["token","contribution"]),
                             use_container_width=True, hide_index=True)
            with cC:
                st.markdown("**Delta (prédit − vrai)**")
                wi = clf.coef_[_class_index(pred_label)] - clf.coef_[_class_index(true_label)]
                contrib = Xrow.multiply(wi).tocsr()
                word_names = feats.transformer_list[0][1].get_feature_names_out()
                n_word=len(word_names)
                items=[]
                for j,v in zip(contrib.indices, contrib.data):
                    if j < n_word:
                        items.append((word_names[j], float(v)))
                items.sort(key=lambda x:x[1], reverse=True)
                st.dataframe(_pd.DataFrame(items[:topk], columns=["token","delta"]),
                             use_container_width=True, hide_index=True)


# ====================================================================================================================
# SECTION 6.b: TRANSFORMER FRANÇAIS
# ====================================================================================================================
elif choice == "8. Transformer Français":
    st.header("Transformer camemBERT")

    import pandas as _pd
    import plotly.express as _px

    # # Lottie (optionnel)
    # try:
    #     from streamlit_lottie import st_lottie  # type: ignore
    #     import requests
    #     r = requests.get("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json", timeout=3)
    #     if r.status_code == 200:
    #         st_lottie(r.json(), height=160, key="lottie_fr")
    # except Exception:
    #     pass

    st.markdown(f"""
    <div style=\"background: linear-gradient(135deg, rgba(0,51,102,0.08), rgba(217,130,43,0.10));
                padding: 18px 18px; border-radius: 14px; border: 1px solid rgba(0,0,0,0.06);\">
      <h2 style=\"margin:0;color:{COLORS['primary']};\">Stratégie de prétraitement appliquée aux textes</h2>
      <p style=\"margin:6px 0 0 0; color:#334; font-size: 0.98em;\">
        Les stratégies évaluées couvrent un spectre progressif allant du texte brut à des approches plus élaborées afin d’analyser finement l’apport de chaque niveau de transformation.</b>
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Graphique 1 — Ablation preprocess (CamemBERT-base)
    st.subheader("Ablation preprocess avec CamemBERT-base")
    pre_df = _pd.DataFrame([
        {"strategy":"texte brut","f1_val":0.8961},
        {"strategy":"numtok light","f1_val":0.8958},
        {"strategy":"cleaner","f1_val":0.8944},
        {"strategy":"cleaner + numtok light","f1_val":0.8929},
        {"strategy":"numtok full","f1_val":0.8913},
    ]).sort_values("f1_val", ascending=False)

    _tmp = pre_df.assign(rank=range(1, len(pre_df) + 1)).copy()
    _tmp["label"] = _tmp["strategy"].map(lambda s: f"<b>{s}</b>")
    fig = _px.scatter(
        _tmp,
        x="rank",
        y="f1_val",
        text="label",
        title="Écarts faibles entre stratégies de prétraitement",
    )
    fig.update_traces(
        textposition="top center",
        marker=dict(size=18),
        textfont=dict(size=18)
    )
    fig.update_layout(
        xaxis_title="",
        yaxis_title="F1 weighted (val)",
        height=440,
        margin=dict(t=70)
    )
    fig.update_yaxes(range=[float(_tmp["f1_val"].min()) - 0.002, float(_tmp["f1_val"].max()) + 0.004])
    st.plotly_chart(fig, use_container_width=True)


    st.markdown("---")

    # Graphique 2 — Le vrai levier : modèle large
    st.subheader("CamemBERT-base vs CamemBERT-large")
    large_df = _pd.DataFrame([
        {"model":"CamemBERT-base (brut)","f1_val":0.8961},
        {"model":"CamemBERT-large (brut)","f1_val":0.9096},
        {"model":"CamemBERT-large (numtok light)","f1_val":0.9106},
    ]).sort_values("f1_val", ascending=False)

    # Version retenue : v2
    fig = _px.line(
        large_df.sort_values("f1_val"),
        x="model",
        y="f1_val",
        markers=True,
        title="Progression base → large"
    )
    fig.update_traces(marker=dict(size=12), line=dict(width=3))
    fig.update_layout(xaxis_title="", yaxis_title="F1 weighted (val)", height=420)
    st.plotly_chart(fig, use_container_width=True)


    st.markdown("---")

    # Microscope preprocess
    st.subheader("Analyse intéractive des stratégies de prétraitement")
    sample = st.text_area("Texte produit", value="iPhone 12 64GB - Coque silicone rouge, 12cm, neuf", height=90)

    import re as _re
    _UNIT_RE = _re.compile(r"(?i)\b(cm|mm|m|kg|g|mg|l|ml|cl|gb|tb|hz|w|kw|v|mah|mp|px|inch|in)\b")

    def _cleaner_basic(t: str) -> str:
        t = (t or "").lower()
        t = _re.sub(r"<[^>]+>", " ", t)
        t = _re.sub(r"[^\w\s-]", " ", t)
        t = _re.sub(r"\s+", " ", t).strip()
        return t

    def _numtok_light(t: str) -> str:
        t = t or ""
        t = replace_numeric_expressions(t, mode='light')
        return t

    a,b,c = st.columns(3)
    with a:
        st.markdown("**Brut**")
        st.code(sample, language="text")
    with b:
        st.markdown("**Cleaner (léger)**")
        st.code(_cleaner_basic(sample), language="text")
    with c:
        st.markdown("**NumTok light**")
        st.code(_numtok_light(sample), language="text")

    # st.info("On suit principalement le **F1 weighted** : classes déséquilibrées, métrique représentative du trafic réel.")



    st.markdown("---")


    st.markdown("""
    <div style="background:#eaf7ee;border:1px solid rgba(0,0,0,0.08);padding:14px;border-radius:14px;">
      <h4 style="margin:0;color:#1b5e20;">✅ Choix retenu</h4>
      <p style="margin:6px 0 0 0;color:#2b2b2b;">
        <b>CamemBERT-large + numtok light</b> — <b>F1 weighted = 0.9106</b>
      </p>
    </div>
    """, unsafe_allow_html=True)


# ====================================================================================================================
# SECTION 6.c: TRANSFORMER MULTILINGUE (ROUTEUR) — epoch 5 + timeline en bas
# ====================================================================================================================
elif choice == "9. Transformer Multilingue":
    st.header("9. Transformer Multilingue")

    import pandas as _pd
    import plotly.express as _px

    st.markdown(f"""
    <div style=\"background: linear-gradient(135deg, rgba(0,51,102,0.08), rgba(217,130,43,0.10));
                padding: 18px 18px; border-radius: 14px; border: 1px solid rgba(0,0,0,0.06);\">
      <h2 style=\"margin:0;color:{COLORS['primary']};\">Evaluation de XLM-RoBERTa-base</h2>
      <p style=\"margin:6px 0 0 0; color:#334; font-size: 0.98em;\">
        Notre objectif est d'explorer une architecture complémentaire en vue d’une stratégie de fusion de modèles.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Composants interactifs (optionnels)
    _HAS_TIMELINE = False
    _HAS_ECHARTS = False
    try:
        from streamlit_timeline import st_timeline  # type: ignore
        _HAS_TIMELINE = True
    except Exception:
        pass
    try:
        from streamlit_echarts import st_echarts  # type: ignore
        _HAS_ECHARTS = True
    except Exception:
        pass

    st.markdown("---")

    # Courbe d'apprentissage XLM-R (v1..v5)
    st.subheader("Courbe d'apprentissage XLM-RoBERTa-base")
    curve = _pd.DataFrame([
        {"epoch":1,"val_f1":0.846235,"val_loss":0.621},
        {"epoch":2,"val_f1":0.861120,"val_loss":0.581},
        {"epoch":3,"val_f1":0.872540,"val_loss":0.553},
        {"epoch":4,"val_f1":0.879930,"val_loss":0.536},
        {"epoch":5,"val_f1":0.883900,"val_loss":0.528},
        {"epoch":6,"val_f1":0.884900,"val_loss":0.526},
        {"epoch":7,"val_f1":0.885200,"val_loss":0.525},
        {"epoch":8,"val_f1":0.885703,"val_loss":0.526},
    ])

    # Version retenue : v1
    fig = _px.line(curve, x="epoch", y="val_f1", markers=True, title="")
    fig.update_traces(marker=dict(size=12), line=dict(width=3))
    fig.update_layout(height=420, xaxis_title="Epoch", yaxis_title="F1 weighted (val)")
    st.plotly_chart(fig, use_container_width=True)


    st.markdown("---")

    # Comparaison FR vs Multilingue (v1..v5) — v3 demandé
    st.subheader("FR vs Multilingue : comparaison dees modèles")
    comp = _pd.DataFrame([
        {"subset":"FR","model":"CamemBERT","f1":0.917,"logloss":0.368},
        {"subset":"FR","model":"XLM-R","f1":0.894,"logloss":0.364},
        {"subset":"Multilingue","model":"CamemBERT","f1":0.889,"logloss":0.543},
        {"subset":"Multilingue","model":"XLM-R","f1":0.855,"logloss":0.535},
    ])

    tab_bar, tab_dashboard = st.tabs(["Comparaison F1 weighted", "Détails des résultats"])

    with tab_bar:
        st.plotly_chart(
            _px.bar(comp, x="f1", y="model", color="subset", barmode="group", title="F1 weighted"),
            use_container_width=True
        )

    with tab_dashboard:
        fr = comp[comp["subset"]=="FR"].set_index("model")
        ml = comp[comp["subset"]=="Multilingue"].set_index("model")
        a,b = st.columns(2)
        with a:
            st.markdown("### FR")
            c1,c2 = st.columns(2)
            with c1:
                st.metric("CamemBERT F1w", f"{fr.loc['CamemBERT','f1']:.3f}")
                st.metric("CamemBERT LogLoss", f"{fr.loc['CamemBERT','logloss']:.3f}")
            with c2:
                st.metric("XLM-R F1w", f"{fr.loc['XLM-R','f1']:.3f}")
                st.metric("XLM-R LogLoss", f"{fr.loc['XLM-R','logloss']:.3f}")
        with b:
            st.markdown("### Multilingue")
            c1,c2 = st.columns(2)
            with c1:
                st.metric("CamemBERT F1w", f"{ml.loc['CamemBERT','f1']:.3f}")
                st.metric("CamemBERT LogLoss", f"{ml.loc['CamemBERT','logloss']:.3f}")
            with c2:
                st.metric("XLM-R F1w", f"{ml.loc['XLM-R','f1']:.3f}")
                st.metric("XLM-R LogLoss", f"{ml.loc['XLM-R','logloss']:.3f}")

        # st.markdown("""
        # <div style="background:#f8f9fa;border:1px solid rgba(0,0,0,0.06);padding:14px;border-radius:14px;">
        #   <h4 style="margin:0;color:#003366;">Pourquoi LogLoss compte ici ?</h4>
        #   <p style="margin:6px 0 0 0;color:#333;">
            # En routage, la <b>fiabilité</b> des scores est clé : logloss plus faible → scores plus calibrés,
            # utile pour seuils, abstention, fusion.
        #   </p>
        # </div>
        # """, unsafe_allow_html=True)


    st.markdown("---")

    

#     # Workflow (refait à partir du notebook text_06_transformer_multilingue.ipynb)
#     st.subheader("Workflow multilingue : entraînement → sélection epoch5 → routage")

#     st.markdown("""
#     - **Offline (sélection)** : entraînement XLM-R, suivi des métriques par epoch, sélection **epoch 5 (best loss)**, export des probabilités.
#     - **Analyse** : découpe **FR / Multilingue** (langdetect) + comparaison **F1 weighted** & **LogLoss**.
#     - **Online (prod)** : routage simple **FR → CamemBERT-large**, **Multilingue → XLM-R epoch 5**.
#     """)

#     router_dot = r"""
#     digraph R {
#       rankdir=LR;
#       node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10, color="#003366"];
#       edge [color="#333333", arrowsize=0.8];

#       subgraph cluster_offline {
#         label="OFFLINE (sélection)";
#         color="#d9822b"; style="rounded"; fontsize=11; fontcolor="#003366";

#         train [label="Train XLM-R
# (max_len=384, bs=64, lr=2e-5
# + warmup 0.1)", fillcolor="#ffffff"];
#         curves [label="Courbe val_f1 / val_loss
# par epoch", fillcolor="#ffffff"];
#         pick [label="Sélection : epoch 5
# (best val_loss)", fillcolor="#ffffff"];
#         export [label="Export probas
# val/test (npy)", fillcolor="#ffffff"];

#         train -> curves -> pick -> export;
#       }

#       subgraph cluster_analysis {
#         label="ANALYSE";
#         color="#d9822b"; style="rounded"; fontsize=11; fontcolor="#003366";

#         lang [label="Langdetect
# FR vs Multilingue", fillcolor="#ffffff"];
#         metrics [label="F1 weighted + LogLoss
# par sous-ensemble", fillcolor="#ffffff"];
#         delta [label="Analyse par catégorie
# (deltas)", fillcolor="#ffffff"];

#         export -> lang -> metrics -> delta;
#       }

#       subgraph cluster_online {
#         label="ONLINE (prod)";
#         color="#d9822b"; style="rounded"; fontsize=11; fontcolor="#003366";

#         in  [label="Texte produit", fillcolor="#ffffff"];
#         gate [label="Langdetect", fillcolor="#ffffff"];
#         fr  [label="Route FR", fillcolor="#eaf7ee", color="#1b5e20"];
#         ml  [label="Route Multilingue", fillcolor="#fff7e6", color="#d9822b"];
#         cam [label="CamemBERT-large
# (numtok light)", fillcolor="#ffffff"];
#         xlm [label="XLM-R
# (epoch 5)", fillcolor="#ffffff"];
#         out [label="Prédiction code + score", fillcolor="#ffffff"];

#         in -> gate;
#         gate -> fr [label="fr"];
#         gate -> ml [label="≠ fr"];
#         fr -> cam -> out;
#         ml -> xlm -> out;
#       }

#       metrics -> gate [style=dashed, label="règle de routage"];
#     }
#     """

#     st.graphviz_chart(router_dot, use_container_width=True)

#     st.caption("Routage défendable : CamemBERT est le spécialiste FR; XLM-R apporte une robustesse hors FR et une meilleure fiabilité des scores (logloss) utile en production.")

#     st.markdown("---")


    st.markdown("""
    <div style="background:#eaf7ee;border:1px solid rgba(0,0,0,0.08);padding:14px;border-radius:14px;">
      <h4 style="margin:0;color:#1b5e20;">✅ Choix retenu</h4>
      <p style="margin:6px 0 0 0;color:#2b2b2b;">
        <b>CamemBERT-large + XLM-R</b><br>
        <b>XLM-RoBERTa-base est local et complémentaire à CamemBERT-large</b> 
    </div>
    """, unsafe_allow_html=True)


elif choice == "10. Sélection du modèle optimal":
    st.header("10. Sélection du modèle optimal")
    
    st.info("""
    **Analyse comparative des modèles**
    
    Cette section présente l'analyse détaillée de 5 modèles d'apprentissage profond
    entraînés sur le dataset Rakuten. 
    
    **Métrique principale:** F1 Score Weighted (adapté aux classes déséquilibrées)
    
    **Modèles analysés:** LeNet, ResNet50, Vision Transformer, Swin Transformer, ConvNeXt
    """)
    
    
    # Essayer d'importer et exécuter
    try:
        from modules.modeles_drive import analyse_modeles_local
        analyse_modeles_local()
        
    except ImportError:
        st.error("""
        ❌ **Module non trouvé**
        
        Le fichier `modules/modeles_drive.py` est requis.
        
        **Structure attendue:**
        ```
        DS_rakuten/
        ├── app.py
        └── modules/
            ├── preprocessing.py
            ├── dataviz.py
            └── modeles_drive.py    # doit exister
        ```
        """)
        
    except Exception as e:
        st.error(f"❌ Erreur d'exécution: {str(e)[:200]}")
        st.info("Vérifiez votre connexion internet et réessayez.")

# ==================================================================================================================================
# SECTION 8: ESSAI DU MODELE
# ==================================================================================================================================
elif choice == "11. Essai du modèle":

    st.header("11. Essai du modèle")
    st.markdown(
        """
        Cette section propose une **démonstration du modèle final** sur un nouveau produit,
        à partir d'informations saisies manuellement, comme dans un contexte réel.
        """
    )

    # --------------------------------------------------------------------------------------------------
    # Entrées utilisateur
    # --------------------------------------------------------------------------------------------------
    st.subheader("Entrées du produit")

    designation = st.text_input(
        "Désignation du produit (obligatoire)"
    )

    description = st.text_area(
        "Description du produit (optionnelle)"
    )

    uploaded_image = st.file_uploader(
        "Image du produit (obligatoire)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        st.subheader("Image du produit")
        st.image(
            uploaded_image,
            caption="Image uploadée",
            width=250,
        )


    # --------------------------------------------------------------------------------------------------
    # Bouton de prédiction
    # --------------------------------------------------------------------------------------------------
    if st.button("Lancer la prédiction"):

        if designation.strip() == "":
            st.error("La désignation du produit est obligatoire.")
            st.stop()

        if uploaded_image is None:
            st.error("Une image est obligatoire pour lancer la prédiction.")
            st.stop()



        # --------------------------------------------------------------------------------------------------
        # Génération des identifiants
        # --------------------------------------------------------------------------------------------------
        product_id = uuid.uuid4().int % 10**8
        image_id = uuid.uuid4().int % 10**8

        # --------------------------------------------------------------------------------------------------
        # Gestion et sauvegarde de l'image (si fournie)
        # --------------------------------------------------------------------------------------------------
        image_path = None
        IMAGE_DIR = Path(str(IMAGE_DEMO_DIR))
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)

        if uploaded_image is not None:
            img = Image.open(uploaded_image).convert("RGB")

            # Resize avec padding blanc (500x500)
            target_size = 500
            img.thumbnail((target_size, target_size))

            background = Image.new("RGB", (target_size, target_size), (255, 255, 255))
            offset = (
                (target_size - img.size[0]) // 2,
                (target_size - img.size[1]) // 2,
            )
            background.paste(img, offset)

            image_filename = f"image_{image_id}_product_{product_id}.jpg"
            image_path = IMAGE_DIR / image_filename
            background.save(image_path)


        # --------------------------------------------------------------------------------------------------
        # Construction d'une ligne DataFrame compatible avec le pipeline
        # --------------------------------------------------------------------------------------------------
        df_demo = pd.DataFrame(
            [
                {
                    "productid": product_id,
                    "imageid": image_id,
                    "designation": designation,
                    "description": description,
                }
            ]
        )

        # --------------------------------------------------------------------------------------------------
        # Prédiction avec contributions
        # --------------------------------------------------------------------------------------------------
        with st.spinner("Prédiction en cours..."):
            result = model.predict_with_contributions(df_demo)
            result['category_name'] = result['label_pred'].map(CATEGORY_NAMES)

        # --------------------------------------------------------------------------------------------------
        # Affichage des résultats
        # --------------------------------------------------------------------------------------------------

        st.subheader("Résultat de la prédiction")

        st.markdown(
            f"""
            **Code catégorie :** `{result.loc[0, "label_pred"]}`  
            **Libellé catégorie :** **{result.loc[0, "category_name"]}**
            """
        )

        st.metric(
            "Confiance globale (méta-modèle)",
            f"{result.loc[0, 'P_final']:.2f}"
        )

        p_final = float(result.loc[0, "P_final"])
        st.caption(f"Probabilité finale après fusion : **{p_final:.2f}**")

        st.subheader("Contributions par modalité")

        p_text = float(result.loc[0, "P_text"])
        p_image = float(result.loc[0, "P_image"])

        st.progress(p_text)
        st.caption(f"Contribution du texte : **{p_text:.2f}**")

        st.progress(p_image)
        st.caption(f"Contribution de l'image : **{p_image:.2f}**")

        # --------------------------------------------------------------------------------------------------
        # Message méthodologique
        # --------------------------------------------------------------------------------------------------
        st.info(
            "La prédiction est réalisée via le même pipeline multimodal "
            "que celui utilisé pour l’évaluation finale du modèle."
        )

        st.info(
            "Les valeurs affichées correspondent aux probabilités associées "
            "à la classe prédite, estimées séparément par le modèle texte, "
            "le modèle image et le méta-modèle de fusion."
        )


# ==================================================================================================================================
# SECTION 9: CONCLUSIONS ET PERSPECTIVES 
# ==================================================================================================================================
elif choice == "12. Conclusions et perspectives":
    st.header("12. Conclusions et perspectives")

    # Objectif
    st.subheader("🎯 Objectif du projet")
    st.markdown(
        """
        Classification automatique de produits e-commerce à partir de **données textuelles et visuelles**
        dans le cadre du challenge Rakuten, avec une analyse fine de la contribution de chaque modalité.
        """
    )

    # Résultats clés
    st.subheader("🔑 Résultats clés")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **Texte**
            - Signal principal pour la majorité des catégories  
            - Excellentes performances avec **CamemBERT**  
            - Probabilités mieux calibrées via combinaison de modèles
            """
        )

    with col2:
        st.markdown(
            """
            **Image**
            - Signal complémentaire mais informatif  
            - Meilleures performances avec **ConvNeXt** et **Swin Transformer**  
            - Robustesse accrue via fusion de modèles visuels
            """
        )

    # Fusion multimodale
    st.subheader("🔗 Fusion multimodale (aboutissement)")
    st.markdown(
        """
        - **Stacking calibré** des pipelines texte et image  
        - Correction des ambiguïtés textuelles  
        - Aucune dégradation des classes déjà bien maîtrisées  
        - Contribution moyenne : **56 % texte / 44 % image**
        """
    )

    # Limites & perspectives
    st.subheader("🔮 Limites et perspectives")
    st.markdown(
        """
        - Catégories encore difficiles : *Jeux éducatifs*, *Jeux de rôle*, *Jouets & Figurines*  
        - Enrichissement des **signaux numériques** (discrétisation plus fine)  
        - **Classification hiérarchique** ou **modèles spécialisés activés conditionnellement**
        """
    )

    # Message de clôture
    st.success(
        "👉 Une approche **multimodale raisonnée, progressive et interprétable**, "
        "constituant une **base solide, performante et extensible** pour la classification de produits à grande échelle."
    )
