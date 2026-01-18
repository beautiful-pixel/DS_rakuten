from pathlib import Path
import base64
from IPython.display import HTML, display



def beautiful_print(txt, style: str = "primary"):
    """
    Affiche un bloc HTML stylisé dans un notebook (Jupyter, Colab, etc.).

    Cette fonction permet de mettre en forme du texte ou du contenu HTML
    afin de structurer visuellement un notebook (titres, messages clés,
    avertissements, résultats intermédiaires, etc.).

    Plusieurs styles prédéfinis sont disponibles pour différencier
    les types de messages (information, succès, avertissement).

    Args:
        txt (str):
            Texte ou contenu HTML à afficher.
        style (str, default="primary"):
            Style visuel à appliquer. Valeurs possibles :
            - "primary"   : bleu clair (information principale)
            - "secondary" : beige / orange clair (information complémentaire)
            - "success"   : vert clair (validation, succès)
            - "warning"   : rouge clair (attention, erreur)

    Raises:
        ValueError:
            Si le style demandé n'existe pas parmi les styles disponibles.

    Returns:
        None:
            La fonction affiche directement le contenu formaté
            et ne retourne aucune valeur.
    """


    styles = {
        "primary": {
            "bg": "#f7fbff",
            "border": "#4682B4",
        },
        "secondary": {
            "bg": "#FFF4E5",
            "border": "#E6A23C",
        },
        "success": {
            "bg": "#F2F7EC",
            "border": "#67C23A",
        },
        "warning": {
            "bg": "#FDECEC",
            "border": "#F56C6C",
        },
    }

    if style not in styles:
        raise ValueError(f"Unknown style '{style}'. Available: {list(styles.keys())}")

    s = styles[style]

    html = f"""
    <div style="
        margin:10px 0;
        padding:12px;
        border-left:5px solid {s['border']};
        background:{s['bg']};
        border-radius:4px;
    ">
        {txt}
    </div>
    """

    display(HTML(html))



def display_df_with_images(
    df,
    image_col: str = "image_path",
    img_width: int = 160,
    max_rows: int | None = None,
):
    """
    Affiche un DataFrame en intégrant le rendu d’images à partir de chemins locaux.

    Cette fonction est conçue pour les notebooks d’exploration de données
    et de vision par ordinateur. Elle convertit une colonne contenant des
    chemins vers des fichiers image locaux en images HTML affichées
    directement dans le tableau.

    Les images sont encodées en base64 afin d’être affichées sans dépendance
    externe au système de fichiers au moment du rendu.

    Args:
        df (pd.DataFrame):
            DataFrame contenant une colonne avec les chemins des images.
        image_col (str, default="image_path"):
            Nom de la colonne contenant les chemins vers les fichiers image.
        img_width (int, default=160):
            Largeur des images affichées, en pixels.
        max_rows (int | None, default=None):
            Nombre maximal de lignes à afficher.
            Si None, toutes les lignes du DataFrame sont affichées.

    Returns:
        None:
            La fonction affiche directement le DataFrame enrichi
            des images et ne retourne aucun objet.
    """

    df_view = df.copy()

    if max_rows is not None:
        df_view = df_view.head(max_rows)

    def _img_to_html(path):
        if not path or not Path(path).exists():
            return ""
        b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        return f'<img src="data:image/jpeg;base64,{b64}" width="{img_width}" />'

    df_view["image"] = df_view[image_col].apply(_img_to_html)

    display(
        HTML(
            df_view.drop(columns=[image_col]).to_html(
                escape=False,
                index=True
            )
        )
    )
