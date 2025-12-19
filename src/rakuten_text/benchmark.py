import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
from .preprocessing import clean_text, get_available_options


def load_dataset(data_dir="../data"):

    X_train = pd.read_csv(f"{data_dir}/X_train_update.csv", index_col=0)
    Y_train = pd.read_csv(f"{data_dir}/Y_train_CVw08PX.csv", index_col=0)

    df = X_train.join(Y_train, how="inner")

    # Créer text_raw : designation + " " + description
    df["text_raw"] = (
        df["designation"].fillna("").astype(str).str.strip() + " " +
        df["description"].fillna("").astype(str).str.strip()
    ).str.strip()

    return df



def define_experiments():
    experiments = []

    experiments.append({
        "name": "baseline_raw",
        "group": "0_Baseline",
        "config": {}  # Toutes les options False par défaut
    })

    experiments.append({
        "name": "fix_encoding",
        "group": "1_Encodage",
        "config": {"fix_encoding": True}
    })

    experiments.append({
        "name": "unescape_html",
        "group": "1_Encodage",
        "config": {"unescape_html": True}
    })

    experiments.append({
        "name": "normalize_unicode",
        "group": "1_Encodage",
        "config": {"normalize_unicode": True}
    })

    experiments.append({
        "name": "all_encoding_fixes",
        "group": "1_Encodage",
        "config": {
            "fix_encoding": True,
            "unescape_html": True,
            "normalize_unicode": True
        }
    })

    experiments.append({
        "name": "remove_html_tags",
        "group": "2_HTML",
        "config": {"remove_html_tags": True}
    })

    experiments.append({
        "name": "remove_boilerplate",
        "group": "2_HTML",
        "config": {"remove_boilerplate": True}
    })

    experiments.append({
        "name": "lowercase",
        "group": "3_Casse",
        "config": {"lowercase": True}
    })

    experiments.append({
        "name": "merge_dimensions",
        "group": "4_Fusions",
        "config": {"merge_dimensions": True}
    })

    experiments.append({
        "name": "merge_units",
        "group": "4_Fusions",
        "config": {"merge_units": True}
    })

    experiments.append({
        "name": "merge_durations",
        "group": "4_Fusions",
        "config": {"merge_durations": True}
    })

    experiments.append({
        "name": "merge_age_ranges",
        "group": "4_Fusions",
        "config": {"merge_age_ranges": True}
    })

    experiments.append({
        "name": "tag_years",
        "group": "4_Fusions",
        "config": {"tag_years": True}
    })

    # Combo : Toutes les fusions structurelles
    experiments.append({
        "name": "all_merges",
        "group": "4_Fusions",
        "config": {
            "merge_dimensions": True,
            "merge_units": True,
            "merge_durations": True,
            "merge_age_ranges": True
        }
    })

    experiments.append({
        "name": "remove_punctuation",
        "group": "5_Ponctuation",
        "config": {"remove_punctuation": True}
    })


    experiments.append({
        "name": "remove_stopwords",
        "group": "6_Filtrage",
        "config": {"remove_stopwords": True}
    })

    experiments.append({
        "name": "remove_single_letters",
        "group": "6_Filtrage",
        "config": {"remove_single_letters": True}
    })

    experiments.append({
        "name": "remove_single_digits",
        "group": "6_Filtrage",
        "config": {"remove_single_digits": True}
    })

    experiments.append({
        "name": "remove_pure_punct_tokens",
        "group": "6_Filtrage",
        "config": {"remove_pure_punct_tokens": True}
    })


    # Approche "clean" traditionnelle
    experiments.append({
        "name": "traditional_cleaning",
        "group": "7_Combos",
        "config": {
            "fix_encoding": True,
            "unescape_html": True,
            "normalize_unicode": True,
            "remove_html_tags": True,
            "lowercase": True,
            "remove_punctuation": True,
            "remove_stopwords": True
        }
    })

    # AB TEST: Traditional sans remove_stopwords
    experiments.append({
        "name": "traditional_without_stopwords",
        "group": "7_Combos",
        "config": {
            "fix_encoding": True,
            "unescape_html": True,
            "normalize_unicode": True,
            "remove_html_tags": True,
            "lowercase": True,
            "remove_punctuation": True,
            # Pas de remove_stopwords pour tester son impact
        }
    })

    # Approche conservatrice (encodage + HTML seulement)
    experiments.append({
        "name": "conservative_cleaning",
        "group": "7_Combos",
        "config": {
            "fix_encoding": True,
            "unescape_html": True,
            "normalize_unicode": True,
            "remove_html_tags": True
        }
    })

    # AB TEST: Lowercase + remove_stopwords
    experiments.append({
        "name": "lowercase_with_stopwords",
        "group": "8_AB_Tests",
        "config": {
            "lowercase": True,
            "remove_stopwords": True
        }
    })

    # AB TEST: Lowercase + remove_punctuation (sans stopwords)
    experiments.append({
        "name": "lowercase_with_punctuation",
        "group": "8_AB_Tests",
        "config": {
            "lowercase": True,
            "remove_punctuation": True
        }
    })

    # AB TEST: Lowercase seul (déjà testé mais regroupé pour comparaison)
    experiments.append({
        "name": "lowercase_only",
        "group": "8_AB_Tests",
        "config": {
            "lowercase": True
        }
    })

    # AB TEST: Encodage + HTML + lowercase (sans ponctuation ni stopwords)
    experiments.append({
        "name": "minimal_cleaning_with_lowercase",
        "group": "8_AB_Tests",
        "config": {
            "fix_encoding": True,
            "unescape_html": True,
            "normalize_unicode": True,
            "remove_html_tags": True,
            "lowercase": True
        }
    })

    # Fusions seulement (pas de suppression)
    experiments.append({
        "name": "merges_only",
        "group": "7_Combos",
        "config": {
            "merge_dimensions": True,
            "merge_units": True,
            "merge_durations": True,
            "merge_age_ranges": True
        }
    })

    return experiments



def run_benchmark(
    df,
    experiments=None,
    test_size=0.15,
    random_state=42,
    tfidf_max_features=10000,
    tfidf_ngram_range=(1, 2),
    verbose=True
):
    if experiments is None:
        experiments = define_experiments()

    if verbose:
        print("=" * 80)
        print("CONFIGURATION DU BENCHMARK")
        print("=" * 80)
        print(f"Total expériences      : {len(experiments)}")
        print(f"Taille de test         : {test_size}")
        print(f"État aléatoire         : {random_state}")
        print(f"TF-IDF max features    : {tfidf_max_features:,}")
        print(f"TF-IDF plage n-grammes : {tfidf_ngram_range}")
        print("=" * 80)
        print()

    # Préparer les labels
    y = df["prdtypecode"].values

    # Créer une seule division train/test (partagée entre toutes les expériences)
    if verbose:
        print("Création de la division train/test...")

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    y_train = y[train_idx]
    y_test = y[test_idx]

    if verbose:
        print(f"  Train : {len(train_idx):,} échantillons")
        print(f"  Test  : {len(test_idx):,} échantillons")
        print()

    # Stocker les résultats
    results = []
    baseline_f1 = None

    # Exécuter les expériences
    for i, exp in enumerate(experiments, 1):
        exp_name = exp["name"]
        exp_group = exp["group"]
        exp_config = exp["config"]

        if verbose:
            print(f"[{i}/{len(experiments)}] {exp_name}")
            print(f"  Groupe : {exp_group}")
            print(f"  Config : {exp_config if exp_config else 'Aucune (données brutes)'}")

        # Appliquer le nettoyage à TOUTES les données d'abord
        if verbose:
            print("  Nettoyage du texte...", end=" ")

        df[f"text_clean_{exp_name}"] = df["text_raw"].apply(
            lambda x: clean_text(x, **exp_config)
        )

        if verbose:
            avg_len = df[f"text_clean_{exp_name}"].str.len().mean()
            print(f"✓ (longueur moyenne : {avg_len:.0f} caractères)")

        # Extraire train/test en utilisant les indices partagés
        X_train_text = df[f"text_clean_{exp_name}"].values[train_idx]
        X_test_text = df[f"text_clean_{exp_name}"].values[test_idx]

        # Construire le pipeline
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=tfidf_max_features,
                ngram_range=tfidf_ngram_range,
                min_df=2,
                max_df=0.95,
                lowercase=False,  # La fonction de nettoyage gère cela
                # 1 + log(tf) replace tf(si un mot apparaît nombreux, son poids n'augmente pas linéair brutalement)
                sublinear_tf=True 
            )),
            ("clf", LogisticRegression(
                C=2.0,
                max_iter=1000,
                random_state=random_state,
                solver="lbfgs"
            ))
        ])

        # Entraîner
        if verbose:
            print("  Entraînement...", end=" ")
        pipeline.fit(X_train_text, y_train)
        if verbose:
            print("✓")

        # Évaluer
        if verbose:
            print("  Évaluation...", end=" ")
        y_pred = pipeline.predict(X_test_text)
        f1 = f1_score(y_test, y_pred, average="weighted")
        acc = accuracy_score(y_test, y_pred)
        if verbose:
            print("✓")

        # Calculer le delta vs baseline
        if exp_name == "baseline_raw":
            baseline_f1 = f1
            delta_f1 = 0.0
            delta_pct = 0.0
        else:
            delta_f1 = f1 - baseline_f1 if baseline_f1 else 0.0
            delta_pct = (delta_f1 / baseline_f1 * 100) if baseline_f1 else 0.0

        if verbose:
            print(f"  → Score F1 : {f1:.6f} | Exactitude : {acc:.4f}", end="")
            if exp_name != "baseline_raw":
                symbol = "🚀" if delta_f1 > 0 else "📉" if delta_f1 < 0 else "➖"
                print(f" | Δ vs baseline : {symbol} {delta_f1:+.6f} ({delta_pct:+.2f}%)")
            else:
                print(" | [BASELINE]")
            print()

        # Stocker le résultat
        results.append({
            "experiment": exp_name,
            "group": exp_group,
            "f1_weighted": f1,
            "accuracy": acc,
            "delta_f1": delta_f1,
            "delta_pct": delta_pct
        })

        # Nettoyer la colonne temporaire pour économiser la mémoire
        df.drop(columns=[f"text_clean_{exp_name}"], inplace=True)

    if verbose:
        print("=" * 80)
        print("✓ BENCHMARK TERMINÉ")
        print("=" * 80)

    # Créer le DataFrame de résultats
    results_df = pd.DataFrame(results)

    return results_df


def analyze_results(results_df, top_n=10):
    print("\n" + "=" * 80)
    print("ANALYSE DES RÉSULTATS DU BENCHMARK")
    print("=" * 80)
    print()

    # Résumé global
    baseline = results_df[results_df["experiment"] == "baseline_raw"].iloc[0]
    print(f"Score F1 Baseline : {baseline['f1_weighted']:.6f}")
    print()

    # Meilleures améliorations
    print(f"🚀 TOP {top_n} AMÉLIORATIONS :")
    print("-" * 80)
    top_improvements = results_df[results_df["experiment"] != "baseline_raw"].nlargest(top_n, "delta_f1")
    for i, row in top_improvements.iterrows():
        print(f"  {row['experiment']:30s} | F1 : {row['f1_weighted']:.6f} | "
              f"Δ : {row['delta_f1']:+.6f} ({row['delta_pct']:+.2f}%) | Groupe : {row['group']}")
    print()

    # Moins bonnes performances
    print(f"📉 TOP {top_n} DÉGRADATIONS :")
    print("-" * 80)
    bottom_performers = results_df[results_df["experiment"] != "baseline_raw"].nsmallest(top_n, "delta_f1")
    for i, row in bottom_performers.iterrows():
        print(f"  {row['experiment']:30s} | F1 : {row['f1_weighted']:.6f} | "
              f"Δ : {row['delta_f1']:+.6f} ({row['delta_pct']:+.2f}%) | Groupe : {row['group']}")
    print()

    # Analyse par groupe
    print("📊 RÉSUMÉ PAR GROUPE :")
    print("-" * 80)
    group_stats = results_df.groupby("group").agg({
        "delta_f1": ["mean", "max", "min"],
        "experiment": "count"
    }).round(6)
    print(group_stats)
    print()

    print("=" * 80)


def save_results(results_df, output_path="results/benchmark_results.csv"):
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"✓ Résultats sauvegardés dans : {output_path}")


def analyze_stopwords_impact(results_df, show_plot=True):
    """
    Analyse l'impact de remove_stopwords dans différents contextes.

    Cette fonction effectue une analyse A/B pour déterminer si remove_stopwords
    améliore ou dégrade la performance dans différentes configurations.
    """
    print("\n" + "="*80)
    print("🔬 ANALYSE A/B : IMPACT DE REMOVE_STOPWORDS")
    print("="*80)

    # 1. Test isolé
    print("\n1️⃣  TEST ISOLÉ")
    print("-"*80)
    if 'remove_stopwords' in results_df['experiment'].values:
        sw_alone = results_df[results_df['experiment'] == 'remove_stopwords'].iloc[0]
        baseline = results_df[results_df['experiment'] == 'baseline_raw'].iloc[0]
        print(f"Remove_stopwords seul  : F1 = {sw_alone['f1_weighted']:.6f} (Δ = {sw_alone['delta_f1']:+.6f})")
        print(f"Baseline (aucun clean) : F1 = {baseline['f1_weighted']:.6f}")
        if sw_alone['delta_f1'] > 0:
            print("✅ Conclusion : remove_stopwords AMÉLIORE la performance en isolation")
        else:
            print("❌ Conclusion : remove_stopwords DÉGRADE la performance en isolation")

    # 2. Comparaison Traditional avec vs sans stopwords
    print("\n2️⃣  TEST DANS PIPELINE COMPLET (Traditional Cleaning)")
    print("-"*80)
    if 'traditional_cleaning' in results_df['experiment'].values and \
       'traditional_without_stopwords' in results_df['experiment'].values:

        with_sw = results_df[results_df['experiment'] == 'traditional_cleaning'].iloc[0]
        without_sw = results_df[results_df['experiment'] == 'traditional_without_stopwords'].iloc[0]

        diff = with_sw['f1_weighted'] - without_sw['f1_weighted']
        diff_pct = (diff / without_sw['f1_weighted']) * 100

        print(f"Avec remove_stopwords  : F1 = {with_sw['f1_weighted']:.6f}")
        print(f"Sans remove_stopwords  : F1 = {without_sw['f1_weighted']:.6f}")
        print(f"Différence             : {diff:+.6f} ({diff_pct:+.2f}%)")

        if abs(diff) < 0.0001:
            print("➖ Conclusion : remove_stopwords n'a PAS d'impact significatif dans le pipeline")
        elif diff > 0:
            print("✅ Conclusion : remove_stopwords AMÉLIORE la performance dans le pipeline complet")
        else:
            print("❌ Conclusion : remove_stopwords DÉGRADE la performance dans le pipeline complet")
    else:
        print("⚠️  Expérience 'traditional_without_stopwords' non trouvée")
        print("    Exécutez le benchmark complet pour obtenir cette comparaison")

    # 3. Combinaisons avec lowercase
    print("\n3️⃣  COMBINAISONS AVEC LOWERCASE")
    print("-"*80)
    lowercase_tests = {
        'lowercase': 'Lowercase seul',
        'lowercase_with_stopwords': 'Lowercase + stopwords',
        'lowercase_with_punctuation': 'Lowercase + punctuation'
    }

    lowercase_results = []
    for exp, label in lowercase_tests.items():
        if exp in results_df['experiment'].values:
            row = results_df[results_df['experiment'] == exp].iloc[0]
            lowercase_results.append({
                'experiment': exp,
                'label': label,
                'f1': row['f1_weighted'],
                'delta': row['delta_f1']
            })
            print(f"{label:30s} : F1 = {row['f1_weighted']:.6f} (Δ = {row['delta_f1']:+.6f})")

    # Calculer l'effet synergique
    if len(lowercase_results) >= 2:
        baseline_lower = next((r['f1'] for r in lowercase_results if r['experiment'] == 'lowercase'), None)
        lower_sw = next((r['f1'] for r in lowercase_results if r['experiment'] == 'lowercase_with_stopwords'), None)

        if baseline_lower and lower_sw:
            synergy = lower_sw - baseline_lower
            print(f"\nEffet synergique (lowercase + stopwords) : {synergy:+.6f}")
            if synergy > 0:
                print("✅ Stopwords renforce l'effet de lowercase")
            else:
                print("❌ Stopwords affaiblit l'effet de lowercase")

    # 4. Résumé et recommandations
    print("\n" + "="*80)
    print("📊 RÉSUMÉ ET RECOMMANDATIONS")
    print("="*80)

    # Collecter les preuves
    evidence = []

    # Preuve 1: Test isolé
    if 'remove_stopwords' in results_df['experiment'].values:
        sw_alone = results_df[results_df['experiment'] == 'remove_stopwords'].iloc[0]
        if sw_alone['delta_f1'] > 0:
            evidence.append(('isolé', '+', sw_alone['delta_f1']))
        else:
            evidence.append(('isolé', '-', sw_alone['delta_f1']))

    # Preuve 2: Pipeline complet
    if 'traditional_cleaning' in results_df['experiment'].values and \
       'traditional_without_stopwords' in results_df['experiment'].values:
        with_sw = results_df[results_df['experiment'] == 'traditional_cleaning'].iloc[0]
        without_sw = results_df[results_df['experiment'] == 'traditional_without_stopwords'].iloc[0]
        diff = with_sw['f1_weighted'] - without_sw['f1_weighted']

        if abs(diff) < 0.0001:
            evidence.append(('pipeline', '=', diff))
        elif diff > 0:
            evidence.append(('pipeline', '+', diff))
        else:
            evidence.append(('pipeline', '-', diff))

    print("\nPREUVES COLLECTÉES :")
    for context, effect, value in evidence:
        symbol = "✅" if effect == '+' else "❌" if effect == '-' else "➖"
        print(f"  {symbol} Contexte {context:10s} : {effect} {abs(value):.6f}")

    # Recommandation finale
    positive_count = sum(1 for _, effect, _ in evidence if effect == '+')
    negative_count = sum(1 for _, effect, _ in evidence if effect == '-')

    print("\n" + "="*80)
    if positive_count > negative_count:
        print("🎯 RECOMMANDATION : UTILISER remove_stopwords")
        print("   → La majorité des tests montrent une amélioration")
    elif negative_count > positive_count:
        print("🎯 RECOMMANDATION : NE PAS UTILISER remove_stopwords")
        print("   → La majorité des tests montrent une dégradation")
    else:
        print("🎯 RECOMMANDATION : EFFET NEUTRE")
        print("   → L'impact de remove_stopwords dépend du contexte")
    print("="*80 + "\n")

    # Visualisation
    if show_plot:
        try:
            import matplotlib.pyplot as plt

            # Graphique de comparaison Traditional
            if 'traditional_cleaning' in results_df['experiment'].values and \
               'traditional_without_stopwords' in results_df['experiment'].values:

                with_sw = results_df[results_df['experiment'] == 'traditional_cleaning'].iloc[0]
                without_sw = results_df[results_df['experiment'] == 'traditional_without_stopwords'].iloc[0]

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

                # Graphique 1: Comparaison directe
                configs = ['Sans stopwords', 'Avec stopwords']
                scores = [without_sw['f1_weighted'], with_sw['f1_weighted']]
                diff = with_sw['f1_weighted'] - without_sw['f1_weighted']
                colors = ['#e74c3c', '#2ecc71'] if diff > 0 else ['#2ecc71', '#e74c3c']

                bars = ax1.bar(configs, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
                ax1.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
                ax1.set_title('Impact de remove_stopwords\ndans Traditional Cleaning',
                             fontsize=13, fontweight='bold')
                ax1.set_ylim([min(scores) - 0.002, max(scores) + 0.003])
                ax1.grid(axis='y', alpha=0.3, linestyle='--')

                for bar, score in zip(bars, scores):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.0003,
                            f'{score:.6f}',
                            ha='center', va='bottom', fontsize=11, fontweight='bold')

                # Graphique 2: Tous les tests A/B
                ab_tests_data = []
                ab_labels = []

                test_configs = [
                    ('baseline_raw', 'Baseline'),
                    ('remove_stopwords', 'Stopwords seul'),
                    ('lowercase', 'Lowercase'),
                    ('lowercase_with_stopwords', 'Lower + SW'),
                    ('traditional_without_stopwords', 'Trad sans SW'),
                    ('traditional_cleaning', 'Trad avec SW')
                ]

                for exp, label in test_configs:
                    if exp in results_df['experiment'].values:
                        score = results_df[results_df['experiment'] == exp].iloc[0]['f1_weighted']
                        ab_tests_data.append(score)
                        ab_labels.append(label)

                if ab_tests_data:
                    colors_ab = plt.cm.RdYlGn([(s - min(ab_tests_data)) / (max(ab_tests_data) - min(ab_tests_data))
                                               for s in ab_tests_data])
                    bars2 = ax2.barh(ab_labels, ab_tests_data, color=colors_ab, edgecolor='black', linewidth=1.5)
                    ax2.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
                    ax2.set_title('Vue d\'ensemble des configurations', fontsize=13, fontweight='bold')
                    ax2.grid(axis='x', alpha=0.3, linestyle='--')

                    for bar, score in zip(bars2, ab_tests_data):
                        width = bar.get_width()
                        ax2.text(width + 0.0002, bar.get_y() + bar.get_height()/2.,
                                f'{score:.6f}',
                                ha='left', va='center', fontsize=9, fontweight='bold')

                plt.tight_layout()
                plt.show()

        except ImportError:
            print("⚠️  matplotlib non disponible pour la visualisation")

    return evidence
