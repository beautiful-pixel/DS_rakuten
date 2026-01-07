import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# ==========================
# Définir une palette 
# ==========================
primary_color = "#003366"   # Bleu foncé pour les titres
secondary_color = "#f2f2f2" # Gris clair pour le fond
accent_color = "#d9822b"    # Orange pour boutons ou accents

# ==========================
# Styles CSS globaux
# ==========================
st.markdown(
    f"""
    <style>
    .main {{
        background-color: {secondary_color};
        color: {primary_color};
    }}
    .sidebar .sidebar-content {{
        background-color: {primary_color};
        color: white;
        font-size:16px;
    }}
    .stButton>button {{
        background-color: {accent_color};
        color: white;
        border-radius: 5px;
        font-weight:bold;
    }}
    .stHeader {{
        color: {primary_color};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================
# Titre et menu latéral
# ==========================
st.title("Projet Data Science - Rakuten")

menu = [
    "Méthodologie",
    "Jeux de données",
    "Préprocessing",
    "DataViz",
    "Features & Modélisation",
    "Sélection du modèle optimal",
    "Conclusions et perspectives"
]

choice = st.sidebar.radio("Sections", menu)
st.sidebar.markdown("---")  # Séparateur 

# ==========================
# Sections du projet
# ==========================
if choice == "Méthodologie":
    with st.container():
        st.header("Méthodologie Data Science")
        st.markdown(
            "Décrivez ici le problème, le contexte et l'insertion du projet dans votre métier."
        )
        st.info("Section placeholder")

elif choice == "Jeux de données":
    with st.container():
        st.header("Jeux de données")
        st.markdown(
            "Description des jeux de données, difficultés, types de variables, etc."
        )
        st.info("Section placeholder")

elif choice == "Préprocessing":
    with st.container():
        st.header("Préprocessing")
        st.markdown("Ici on exécutera les fonctions du module `preprocessing`")
        st.button("Simuler préprocessing")  # Bouton placeholder

elif choice == "DataViz":
    with st.container():
        st.header("Exploration et DataViz")
        st.markdown("Ici on exécutera les fonctions du module `dataviz` pour visualiser les données")
        
        # --------------------------
        # Exemple de graphique placeholder
        # --------------------------
        x = np.random.randn(100)
        fig, ax = plt.subplots()
        ax.hist(x, bins=15, color=primary_color, edgecolor="white")
        ax.set_title("Histogramme exemple", color=primary_color)
        st.pyplot(fig)

elif choice == "Features & Modélisation":
    with st.container():
        st.header("Features & Modélisation")
        st.markdown("Ici on exécutera les fonctions du module `modeling` pour créer et tester les modèles")
        st.button("Simuler modélisation")  # Bouton placeholder

elif choice == "Sélection du modèle optimal":
    with st.container():
        st.header("Sélection et évaluation du modèle optimal")
        st.markdown("Comparaison des modèles et sélection du meilleur")
        st.info("Section placeholder")

elif choice == "Conclusions et perspectives":
    with st.container():
        st.header("Conclusions et perspectives")
        st.markdown(
            "Résumé des résultats et recommandations, pistes pour travaux futurs."
        )
        st.info("Section placeholder")
