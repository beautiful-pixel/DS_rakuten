import matplotlib.pyplot as plt

def images_grid(
    images,
    nrows=5,
    ncols=10,
    cmap=None,
    titles=None,
    ordered_by_rows=True,
    axes_size=(1.5, 1.5),
):
    """
    Affiche une grille d'images.

    Les images sont disposées dans une grille de taille (nrows × ncols),
    avec la possibilité de contrôler l'ordre de remplissage et
    d'ajouter des titres individuels.

    Args:
        images (Sequence[np.ndarray]): Liste ou tableau d'images à afficher.
            Chaque image doit être compatible avec `plt.imshow`.
        nrows (int, optional): Nombre de lignes de la grille. Par défaut 5.
        ncols (int, optional): Nombre de colonnes de la grille. Par défaut 10.
        cmap (str or None, optional): Colormap utilisée pour l'affichage
            (ex. "gray" pour images en niveaux de gris). Par défaut None.
        titles (Sequence[str] or None, optional): Titres associés à chaque image.
            Doit être de même longueur que `images`. Par défaut None.
        ordered_by_rows (bool, optional): Si True, remplit la grille ligne par ligne.
            Sinon, remplit colonne par colonne. Par défaut True.
        axes_size (tuple[float, float], optional): Taille (largeur, hauteur)
            de chaque sous-figure en pouces. Par défaut (1.5, 1.5).

    Returns:
        None: La fonction affiche directement la figure.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=(axes_size[0]*ncols, axes_size[1]*nrows))
    axes = axes.flatten()
    if ordered_by_rows:
        axes_order = range(nrows*ncols)
    else:
        axes_order = [i*ncols + j for j in range(ncols) for i in range(nrows)]
    for i in range(nrows*ncols):
        k = axes_order[i]
        if i < len(images):
            axes[k].imshow(images[i], cmap=cmap)
            if titles is not None and i < len(titles):
                axes[k].set_title(titles[i], fontsize=8)
        axes[k].set_xticks([])
        axes[k].set_yticks([])
    plt.show()