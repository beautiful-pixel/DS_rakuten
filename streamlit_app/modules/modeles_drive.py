"""
Module d'analyse des modèles ML - Version FINALE avec F1 Weighted
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import tempfile
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


MODEL_URLS = {
    "lenet_canonical": {
        "name": "LeNet",
        "type": "image",
        "npz_path": "artifacts/lenet_val.npz",
        "color": "#FF6B6B"
    },
    "resnet50_canonical": {
        "name": "ResNet50",
        "type": "image",
        "npz_path": "artifacts/resnet50_val.npz",
        "color": "#4ECDC4"
    },
    "vit_canonical": {
        "name": "Vision Transformer",
        "type": "image",
        "npz_path": "artifacts/vit_val.npz",
        "color": "#45B7D1"
    },
    "swin_canonical": {
        "name": "Swin Transformer",
        "type": "image",
        "npz_path": "artifacts/swin_val.npz",
        "color": "#96CEB4"
    },
    "convnext_canonical": {
        "name": "ConvNeXt",
        "type": "image",
        "npz_path": "artifacts/convnext_val.npz",
        "color": "#FFEAA7"
    }
}


@st.cache_data(show_spinner=True)
def load_model_data_correct(model_key):
    """Charge les données d'un modèle depuis un fichier local .npz."""
    if model_key not in MODEL_URLS:
        return None

    try:
        npz_path = Path(MODEL_URLS[model_key]["npz_path"])

        if not npz_path.exists():
            st.warning(f"Fichier introuvable : {npz_path}")
            return None

        # Charger le fichier .npz
        data = np.load(npz_path)
        file_keys = list(data.keys())

        data_dict = {
            "model_name": MODEL_URLS[model_key]["name"],
            "model_key": model_key,
            "color": MODEL_URLS[model_key]["color"],
            "file_keys": file_keys,
        }

        # 1. y_true
        if "y_true" not in data:
            st.warning(f"'y_true' manquant dans {npz_path.name}")
            return None
        data_dict["y_true"] = data["y_true"]

        # 2. Probabilités
        if "probs" in data:
            probabilities = data["probs"]
        elif "logits" in data:
            probabilities = softmax(data["logits"], axis=1)
        else:
            st.warning(f"'probs' ou 'logits' manquant dans {npz_path.name}")
            return None

        # 3. Prédictions
        data_dict["y_pred"] = np.argmax(probabilities, axis=1)

        # 4. Classes
        if "classes" in data:
            data_dict["classes"] = data["classes"]
        else:
            data_dict["classes"] = np.arange(probabilities.shape[1])

        data_dict["n_samples"] = len(data_dict["y_true"])
        data_dict["n_classes"] = len(data_dict["classes"])

        return data_dict

    except Exception as e:
        st.error(f"Erreur chargement modèle {model_key} : {e}")
        return None


def softmax(x, axis=None):
    """Fonction softmax pour convertir logits en probabilités."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def compute_metrics_correct(data):
    """Calcule les métriques avec F1 Weighted."""
    try:
        y_true = data['y_true']
        y_pred = data['y_pred']
        
        # Vérifier les shapes
        if len(y_true) != len(y_pred):
            return None
        
        # Calculer métriques - F1 WEIGHTED (meilleur pour classes déséquilibrées)
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'n_samples': len(y_true),
            'n_classes': data['n_classes']
        }
        
        return metrics
        
    except Exception as e:
        return None

def plot_simple_confusion_matrix(y_true, y_pred, model_name, max_classes=10):
    """Crée une matrice de confusion simple et robuste."""
    # Limiter le nombre de classes
    n_classes = min(max_classes, len(np.unique(y_true)))
    
    # Filtrer les données
    mask = (y_true < n_classes) & (y_pred < n_classes)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    if len(y_true_filtered) == 0:
        return None
    
    # Calculer la matrice
    cm = confusion_matrix(y_true_filtered, y_pred_filtered)
    
    # Créer la figure
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f"C{i}" for i in range(n_classes)],
        y=[f"C{i}" for i in range(n_classes)],
        colorscale='Blues',
        text=cm.astype(str),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverinfo='z'
    ))
    
    fig.update_layout(
        title=f'Matrice de confusion - {model_name}',
        xaxis_title='Prédit',
        yaxis_title='Réel',
        height=500,
        width=600
    )
    
    return fig

def plot_confusion_matrix_advanced(y_true, y_pred, model_name):
    """Crée une matrice de confusion interactive avec 3 visualisations."""
    
    # Crear pestañas para différentes visualisations
    tab1, tab2, tab3 = st.tabs([
        "Normalisée par ligne (Recall)", 
        "Normalisée par colonne (Précision)", 
        "Comptes bruts"
    ])
    
    # Nombre de classes
    max_classes = 27
    n_classes = min(max_classes, len(np.unique(y_true)))
    
    # Aviso si estamos limitando
    if n_classes < len(np.unique(y_true)):
        st.info(f" Affichage limité à {n_classes} classes sur {len(np.unique(y_true))} pour lisibilité")
    
    mask = (y_true < n_classes) & (y_pred < n_classes)
    y_true_filt = y_true[mask]
    y_pred_filt = y_pred[mask]
    
    if len(y_true_filt) == 0:
        st.info("Pas assez de données pour la matrice de confusion")
        return
    
    # Matriz base
    cm = confusion_matrix(y_true_filt, y_pred_filt)
    
    with tab1:
        # Normalizada por fila (Recall)
        cm_row = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_row = np.nan_to_num(cm_row)
        
        fig = px.imshow(
            cm_row,
            labels=dict(x="Prédit", y="Réel", color="Recall"),
            x=[f"C{i}" for i in range(n_classes)],
            y=[f"C{i}" for i in range(n_classes)],
            title=f"{model_name} - Matrice normalisée par ligne (Recall)",
            color_continuous_scale='Blues',
            aspect="auto",
            text_auto='.0%'
        )
        
        fig.update_traces(
            hovertemplate="<b>Réel: C%{y}</b><br>Prédit: C%{x}<br>Recall: %{z:.1%}<br>Comptes: %{customdata}",
            customdata=cm
        )
        
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas
        recalls = np.diag(cm_row)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Recall moyen", f"{np.mean(recalls):.1%}")
        with col2:
            st.metric("Recall min", f"{np.min(recalls):.1%}")
    
    with tab2:
        # Normalizada por columna (Précision)
        cm_col = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        cm_col = np.nan_to_num(cm_col)
        
        fig = px.imshow(
            cm_col,
            labels=dict(x="Prédit", y="Réel", color="Précision"),
            x=[f"C{i}" for i in range(n_classes)],
            y=[f"C{i}" for i in range(n_classes)],
            title=f"{model_name} - Matrice normalisée par colonne (Précision)",
            color_continuous_scale='Greens',
            aspect="auto",
            text_auto='.0%'
        )
        
        fig.update_traces(
            hovertemplate="<b>Réel: C%{y}</b><br>Prédit: C%{x}<br>Précision: %{z:.1%}<br>Comptes: %{customdata}",
            customdata=cm
        )
        
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas
        precisions = np.diag(cm_col)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Précision moyenne", f"{np.mean(precisions):.1%}")
        with col2:
            st.metric("Précision min", f"{np.min(precisions):.1%}")
    
    with tab3:
        # Bruta (comptes)
        fig = px.imshow(
            cm,
            labels=dict(x="Prédit", y="Réel", color="Comptes"),
            x=[f"C{i}" for i in range(n_classes)],
            y=[f"C{i}" for i in range(n_classes)],
            title=f"{model_name} - Matrice de confusion brute",
            color_continuous_scale='Reds',
            aspect="auto",
            text_auto=True
        )
        
        fig.update_traces(
            hovertemplate="<b>Réel: C%{y}</b><br>Prédit: C%{x}<br>Nombre: %{z}"
        )
        
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas
        accuracy = np.trace(cm) / np.sum(cm)
        st.metric("Accuracy", f"{accuracy:.1%}")

def analyse_modeles_local():
    """Fonction principale avec F1 Weighted."""
    st.header(" Analyse Comparative des Modèles")
    
    
    st.success(f"""
     **5 modèles chargés avec succès**
    
    **Métrique principale: F1 Score Weighted** (recommandé pour classes déséquilibrées)
    """)
    
    # Charger modèles
    with st.spinner("Chargement des modèles en cours..."):
        all_data = {}
        all_metrics = {}
        
        for model_key in MODEL_URLS.keys():
            data = load_model_data_correct(model_key)
            if data:
                all_data[model_key] = data
                metrics = compute_metrics_correct(data)
                if metrics:
                    all_metrics[model_key] = metrics
    
    if not all_data:
        st.error(" Aucun modèle chargé")
        return
    
    # Onglets
    tab1, tab2, tab3 = st.tabs([" **Classement**", "**Analyse**", "**Recommandation**"])
    
    with tab1:
        # Préparer données de classement
        ranking_data = []
        
        for model_key, data in all_data.items():
            if model_key in all_metrics:
                metrics = all_metrics[model_key]
                
                ranking_data.append({
                    'Modèle': data['model_name'],
                    'Accuracy': metrics['accuracy'],
                    'F1 Score (Weighted)': metrics['f1_weighted'],  # F1 Weighted!
                    'Precision (Weighted)': metrics['precision_weighted'],
                    'Recall (Weighted)': metrics['recall_weighted'],
                    'F1 Score (Macro)': metrics['f1_macro'],  # Opcional: mostrar ambos
                    'Échantillons': metrics['n_samples'],
                    'Couleur': data['color']
                })
        
        df = pd.DataFrame(ranking_data)
        
        # Ordenar por F1 Weighted (métrica principal)
        df = df.sort_values('F1 Score (Weighted)', ascending=False)
        df['Position'] = range(1, len(df) + 1)
        
        #  Podium
        st.subheader(" Podium des modèles (par F1 Score Weighted)")
        
        cols = st.columns(min(5, len(df)))
        
        for i in range(min(5, len(df))):
            with cols[i]:
                model = df.iloc[i]
                
                # Medalla según posición
                medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
                colors = [
                    {'bg': '#FFF8DC', 'border': '#DAA520', 'text': '#8B4513'},
                    {'bg': '#F5F5F5', 'border': '#C0C0C0', 'text': '#505050'},
                    {'bg': '#F5DEB3', 'border': '#CD853F', 'text': '#8B4513'},
                    {'bg': '#E6F3FF', 'border': '#4682B4', 'text': '#2F4F4F'},
                    {'bg': '#E8F8E8', 'border': '#32CD32', 'text': '#006400'}
                ][i]
                
                st.markdown(f"""
                <div style='text-align:center;padding:12px;background:{colors['bg']};
                            border:2px solid {colors['border']};border-radius:8px;margin-bottom:8px'>
                    <div style='font-size:20px'>{medal}</div>
                    <div style='font-weight:bold;color:{colors['text']};margin:5px 0;font-size:14px'>
                        {model["Modèle"]}
                    </div>
                    <div style='font-size:22px;font-weight:bold;color:{colors['text']}'>
                        {model["F1 Score (Weighted)"]:.2%}
                    </div>
                    <div style='font-size:11px;color:#666'>F1 Weighted • Pos.{i+1}</div>
                </div>
                """, unsafe_allow_html=True)
        
        #  Tableau détaillé
        st.subheader(" Métriques détaillées")
        
        display_df = df.copy()
        for col in ['Accuracy', 'F1 Score (Weighted)', 'Precision (Weighted)', 'Recall (Weighted)', 'F1 Score (Macro)']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
        
        st.dataframe(
            display_df[['Position', 'Modèle', 'Accuracy', 'F1 Score (Weighted)', 
                       'Precision (Weighted)', 'Recall (Weighted)', 'Échantillons']],
            use_container_width=True,
            height=400
        )
        
        #  Graphique comparatif
        st.subheader(" Comparaison F1 Score vs Accuracy")
        
        fig = go.Figure()
        
        # Añadir puntos para cada modèle
        for _, row in df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['Accuracy']],
                y=[row['F1 Score (Weighted)']],
                mode='markers+text',
                name=row['Modèle'],
                marker=dict(
                    size=15,
                    color=row['Couleur'],
                    line=dict(width=2, color='DarkSlateGrey')
                ),
                text=[row['Modèle']],
                textposition="top center",
                hoverinfo='text',
                hovertext=f"{row['Modèle']}<br>Accuracy: {row['Accuracy']:.2%}<br>F1 Weighted: {row['F1 Score (Weighted)']:.2%}"
            ))
        
        fig.update_layout(
            title='F1 Score Weighted vs Accuracy par modèle',
            xaxis=dict(
                title='Accuracy',
                tickformat='.0%',
                range=[0, 1]
            ),
            yaxis=dict(
                title='F1 Score Weighted',
                tickformat='.0%',
                range=[0, 1]
            ),
            showlegend=False,
            height=500,
            plot_bgcolor='rgba(240,240,240,0.8)',
            shapes=[
                # Línea diagonal de igualdad
                dict(
                    type='line',
                    x0=0, x1=1,
                    y0=0, y1=1,
                    line=dict(color='gray', width=1, dash='dash')
                )
            ]
        )
        
        # Añadir anotación explicativa
        fig.add_annotation(
            x=0.5, y=0.9,
            text="⬆ Meilleurs modèles (F1 élevé)",
            showarrow=False,
            font=dict(size=12, color="green")
        )
        
        fig.add_annotation(
            x=0.9, y=0.5,
            text="➡ Bonne accuracy mais F1 moins bon",
            showarrow=False,
            font=dict(size=12, color="orange")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Analyse détaillée
        st.subheader(" Analyse par modèle")
        
        model_options = [(data['model_name'], key) for key, data in all_data.items()]
        selected_model = st.selectbox(
            "Choisir un modèle pour analyse détaillée",
            [opt[0] for opt in model_options]
        )
        
        selected_key = next(key for name, key in model_options if name == selected_model)
        
        if selected_key in all_data and selected_key in all_metrics:
            data = all_data[selected_key]
            metrics = all_metrics[selected_key]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            with col2:
                st.metric("F1 Weighted", f"{metrics['f1_weighted']:.2%}", 
                         delta=f"{(metrics['f1_weighted'] - metrics['accuracy']):.2%}")
            with col3:
                st.metric("Précision", f"{metrics['precision_weighted']:.2%}")
            with col4:
                st.metric("Rappel", f"{metrics['recall_weighted']:.2%}")
            
            # Comparaison F1 Macro vs Weighted
            st.subheader(f" Comparaison F1 Score - {selected_model}")
            
            fig_f1 = go.Figure(data=[
                go.Bar(name='F1 Weighted', 
                      x=['F1 Scores'], 
                      y=[metrics['f1_weighted']],
                      marker_color='#4ECDC4'),
                go.Bar(name='F1 Macro', 
                      x=['F1 Scores'], 
                      y=[metrics['f1_macro']],
                      marker_color='#45B7D1')
            ])
            
            fig_f1.update_layout(
                barmode='group',
                title=f'Comparaison F1 Weighted vs Macro - {selected_model}',
                yaxis=dict(tickformat='.0%', range=[0, 1]),
                showlegend=True,
                height=400
            )
            
            st.plotly_chart(fig_f1, use_container_width=True)
            
            # Explication F1
            with st.expander("ℹ️ Différence entre F1 Weighted et Macro"):
                st.write("""
                **F1 Weighted:** 
                - Prend en compte le déséquilibre des classes
                - Pondere chaque classe par son nombre d'échantillons
                - **Recommandé** pour notre dataset déséquilibré
                
                **F1 Macro:**
                - Traite toutes les clases de manière égale
                - Non pondéré par le nombre d'échantillons
                - Utile si toutes les classes sont importantes
                """)
            
            #  NUEVA SECCIÓN: Matrices de confusion avanzadas
            st.subheader(f" Matrices de confusion - {selected_model}")
            
            # Explicación inicial
            st.info("""
            **Trois visualisations disponibles:**
            1. **Normalisée par ligne (Recall):** Montre la sensibilité par classe réelle
            2. **Normalisée par colonne (Précision):** Montre la fiabilité des prédictions  
            3. **Comptes bruts:** Nombre absolu d'échantillons
            """)
            
            # Llamar a la función de matrices avanzadas
            plot_confusion_matrix_advanced(data['y_true'], data['y_pred'], selected_model)
            
            # Explicación adicional
            with st.expander(" Guide d'interprétation des matrices", expanded=False):
                st.write("""
                **Comment lire les matrices:**
                
                **Pour la matrice normalisée par ligne (Recall):**
                - Chaque ligne représente une **classe réelle**
                - Chaque colonne représente une **classe prédite**
                - Les valeurs sur la **diagonale** montrent le pourcentage de chaque classe correctement identifiée
                - Les valeurs hors diagonale montrent les **confusions** entre classes
                
                **Exemple:** Si la case (ligne 3, colonne 5) = 15%, cela signifie que:
                - 15% des produits **réellement** de classe 3 ont été **prédits** comme classe 5
                - C'est une erreur de classification entre classe 3 et classe 5
                
                **Pour la matrice normalisée par colonne (Précision):**
                - Chaque colonne représente une **classe prédite**
                - Chaque ligne représente une **classe réelle**
                - Les valeurs sur la diagonale montrent la **fiabilité** des prédictions
                
                **Utilité pratique:**
                - **Recall faible** → Le modèle "oublie" certains produits de cette classe
                - **Précision faible** → Le modèle fait trop de "faux positifs" pour cette classe
                - **Diagonale claire** → Le modèle distingue bien les classes
                - **Cases colorées hors diagonale** → Confusions fréquentes entre classes
                """)
    
    with tab3:
        # Recommandation
        st.subheader(" Recommandation finale")
        
        # Trouver le meilleur modèle par F1 Weighted
        best_model_name = None
        best_f1_weighted = 0
        
        for model_key, metrics in all_metrics.items():
            if metrics['f1_weighted'] > best_f1_weighted:
                best_f1_weighted = metrics['f1_weighted']
                best_model_name = all_data[model_key]['model_name']
        
        if best_model_name:
            best_metrics = next(metrics for model_key, metrics in all_metrics.items() 
                               if all_data[model_key]['model_name'] == best_model_name)
            
            st.success(f"""
            **MODÈLE RECOMMANDÉ: {best_model_name}**
            
            **Performance:**
            - **F1 Weighted**: {best_f1_weighted:.2%} (métrique principale)
            - **Accuracy**: {best_metrics['accuracy']:.2%}
            - **Précision**: {best_metrics['precision_weighted']:.2%}
            - **Rappel**: {best_metrics['recall_weighted']:.2%}
            
            **Pour production:** F1 Weighted > 75% → Performance excellente
            """)
            
            # Tableau comparatif
            st.subheader(" Comparaison synthétique")
            
            comparison_data = []
            for model_key, data in all_data.items():
                if model_key in all_metrics:
                    metrics = all_metrics[model_key]
                    comparison_data.append({
                        'Modèle': data['model_name'],
                        'F1 Weighted': f"{metrics['f1_weighted']:.2%}",
                        'Accuracy': f"{metrics['accuracy']:.2%}",
                        'Différence': f"{(metrics['f1_weighted'] - metrics['accuracy']):.2%}",
                        'Recommandation': '✅' if data['model_name'] == best_model_name else
                                         '⚠️' if metrics['f1_weighted'] > 0.7 else
                                         '❌'
                    })
            
            df_comp = pd.DataFrame(comparison_data)
            st.dataframe(df_comp, use_container_width=True)

if __name__ == "__main__":
    analyse_modeles_drive()