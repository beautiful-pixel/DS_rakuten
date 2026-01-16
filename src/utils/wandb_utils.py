import wandb
import pandas as pd

def load_wandb_runs(
    project_path,
    run_names=None,
    states=("finished",),
    filters=None,
):
    """
    Charge les runs d'un projet Weights & Biases avec filtrage optionnel.

    Args:
        project_path (str):
            Chemin du projet W&B au format "entity/project".
        run_names (list[str] | None):
            Liste des noms de runs à récupérer (ex: ["camembert_texte brute"]).
            Si None, tous les runs sont chargés.
        states (tuple[str]):
            États des runs à inclure (par défaut ("finished",)).
        filters (dict | None):
            Filtres W&B avancés (ex: {"config.transformation": "numtok light"}).

    Returns:
        list[wandb.apis.public.Run]:
            Liste des runs W&B correspondants.
    """
    api = wandb.Api()
    runs = api.runs(project_path, filters=filters)

    selected_runs = []
    for run in runs:
        if states and run.state not in states:
            continue
        if run_names and run.name not in run_names:
            continue
        selected_runs.append(run)

    return selected_runs
    
import pandas as pd


def load_wandb_history_df(
    runs,
    keys=("epoch", "train/loss", "train/f1", "val/loss", "val/f1"),
    config_keys=None,
):
    """
    Construit un DataFrame contenant l'historique par epoch des runs W&B.

    Args:
        runs (list[wandb.apis.public.Run]):
            Liste de runs W&B.
        keys (tuple[str]):
            Métriques à récupérer dans l'historique.
        config_keys (list[str] | None):
            Clés optionnelles à extraire depuis run.config
            (ex: ["model", "transformation"]).

    Returns:
        pd.DataFrame:
            DataFrame concaténé contenant l'historique par epoch.
            Contient toujours la colonne :
            - name
            Et optionnellement les champs issus de config_keys.
    """
    rows = []

    for run in runs:
        hist = run.history(keys=list(keys))

        if hist.empty:
            continue

        # Toujours présent
        hist["name"] = run.name

        # Champs optionnels depuis la config
        if config_keys:
            for key in config_keys:
                hist[key] = run.config.get(key)

        rows.append(hist)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def load_wandb_summary_df(runs):
    """
    Construit un DataFrame récapitulatif des runs W&B
    (une ligne par run).

    Args:
        runs (list[wandb.apis.public.Run]):
            Liste de runs W&B.

    Returns:
        pd.DataFrame:
            DataFrame contenant les métriques finales et la config associée.
    """
    rows = []

    for run in runs:
        row = {}

        # Nom du run
        row["name"] = run.name

        # Config (hyperparamètres)
        for k, v in run.config.items():
            if not k.startswith("_"):
                row[k] = v

        # Summary (métriques finales)
        for k, v in run.summary.items():
            if not isinstance(v, dict):
                row[k] = v

        rows.append(row)

    return pd.DataFrame(rows)
