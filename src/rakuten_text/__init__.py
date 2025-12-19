"""
Rakuten text processing utilities.

Phase 1 (Prétraitement de Texte) - TERMINÉE ✅
Phase 2 (Vectorisation et Modélisation) - EN COURS 🚧

Modules principaux :
- preprocessing : Fonctions de nettoyage de texte (clean_text, final_text_cleaner)
- benchmark : Outils de benchmark et expérimentation (load_dataset, run_benchmark)
- features : Extraction de features manuelles (extract_text_features)
- vectorization : Construction de vectoriseurs (build_count_vectorizer, build_tfidf_vectorizer)
- models : Modèles ML et pipelines (get_model, build_full_pipeline)
- experiments : Framework d'expérimentation (run_strategy_comparison, run_hyperparameter_grid)

Fonction de production recommandée : final_text_cleaner()
"""

# Phase 1: Preprocessing
from .preprocessing import (
    final_text_cleaner,
    clean_text,
    get_available_options,
    print_available_options,
    NLTK_STOPWORDS,
    PUNCTUATION,
    BOILERPLATE_PHRASES,
    text_preprocess,
    get_decored_labels,
    complete_txt_preprocess,
)

# Phase 1: Benchmark
from .benchmark import (
    load_dataset,
    define_experiments,
    run_benchmark,
    analyze_results,
    save_results,
)

# Phase 2: Features
from .features import (
    extract_text_features,
    add_text_features,
    get_feature_names,
    get_length_features,
    get_composition_features,
    describe_features,
    print_feature_summary,
)

# Phase 2: Vectorization
from .vectorization import (
    FeatureWeighter,
    build_count_vectorizer,
    build_tfidf_vectorizer,
    build_vectorizer,
    build_split_vectorizer_pipeline,
    build_merged_vectorizer_pipeline,
    get_available_presets,
    print_available_presets,
    get_vectorizer_info,
    compare_vectorizer_configs,
    save_vectorization_config,
    load_vectorization_config,
    get_config_summary,
)

# Phase 2: Models
from .models import (
    get_model,
    get_available_models,
    get_model_info,
    build_full_pipeline,
    evaluate_pipeline,
    train_and_evaluate,
)

# Categories
from .categories import (
    CATEGORY_NAMES,
    CATEGORY_SHORT_NAMES,
    CATEGORY_GROUPS,
    get_category_name,
    get_all_categories,
    get_category_codes,
    get_category_group,
    format_category_label,
    map_categories_in_dataframe,
    get_category_distribution,
    validate_category_code,
    print_category_summary,
)

# Phase 2: Experiments
from .experiments import (
    run_single_experiment,
    run_hyperparameter_grid,
    run_strategy_comparison,
    run_title_weighting_experiment,
    analyze_results as analyze_experiment_results,
    save_experiment_results,
    track_all_scores,
    verify_best_score,
    generate_vectorization_report,
)

__all__ = [
    # Phase 1: Preprocessing
    "final_text_cleaner",
    "clean_text",
    "get_available_options",
    "print_available_options",
    "NLTK_STOPWORDS",
    "PUNCTUATION",
    "BOILERPLATE_PHRASES",
    "text_preprocess",
    "get_decored_labels",
    "complete_txt_preprocess",

    # Phase 1: Benchmark
    "load_dataset",
    "define_experiments",
    "run_benchmark",
    "analyze_results",
    "save_results",

    # Phase 2: Features
    "extract_text_features",
    "add_text_features",
    "get_feature_names",
    "get_length_features",
    "get_composition_features",
    "describe_features",
    "print_feature_summary",

    # Phase 2: Vectorization
    "FeatureWeighter",
    "build_count_vectorizer",
    "build_tfidf_vectorizer",
    "build_vectorizer",
    "build_split_vectorizer_pipeline",
    "build_merged_vectorizer_pipeline",
    "get_available_presets",
    "print_available_presets",
    "get_vectorizer_info",
    "compare_vectorizer_configs",
    "save_vectorization_config",
    "load_vectorization_config",
    "get_config_summary",

    # Phase 2: Models
    "get_model",
    "get_available_models",
    "get_model_info",
    "build_full_pipeline",
    "evaluate_pipeline",
    "train_and_evaluate",

    # Categories
    "CATEGORY_NAMES",
    "CATEGORY_SHORT_NAMES",
    "CATEGORY_GROUPS",
    "get_category_name",
    "get_all_categories",
    "get_category_codes",
    "get_category_group",
    "format_category_label",
    "map_categories_in_dataframe",
    "get_category_distribution",
    "validate_category_code",
    "print_category_summary",

    # Phase 2: Experiments
    "run_single_experiment",
    "run_hyperparameter_grid",
    "run_strategy_comparison",
    "run_title_weighting_experiment",
    "analyze_experiment_results",
    "save_experiment_results",
    "track_all_scores",
    "verify_best_score",
    "generate_vectorization_report",
]

# Version du module
__version__ = "1.1.0"
__status__ = "Phase 2 - EN COURS"
