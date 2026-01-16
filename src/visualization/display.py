from pathlib import Path
import base64
from IPython.display import HTML, display



def beautiful_print(txt, style: str = "primary"):
    """
    Display a styled HTML block in a notebook.

    Args:
        txt (str): Text or HTML content to display.
        style (str): Style name. Options:
            - "primary"      : bleu clair (information principale)
            - "secondary"    : beige / orange clair (complément)
            - "success"      : vert clair (validation / succès)
            - "warning"      : rouge clair (attention / erreur)
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
    Display a DataFrame with images rendered from a local image path column.

    Args:
        df (pd.DataFrame): DataFrame containing image paths.
        image_col (str): Name of the column with image file paths.
        img_width (int): Width of displayed images in pixels.
        max_rows (int | None): Optional number of rows to display.
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
