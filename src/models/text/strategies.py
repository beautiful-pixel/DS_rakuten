from .pipelines import (
    build_lr_pipeline,
    build_svc_pipeline,
    build_sgd_log_pipeline,
    ...
)

TEXT_STRATEGIES = {
    "S0_LR": build_lr_pipeline,
    "S1_LinearSVC": build_svc_pipeline,
    "S2_SGD_log": build_sgd_log_pipeline,
    "S3_SGD_hinge": build_sgd_hinge_pipeline,
    "S4_ComplementNB": build_nb_pipeline,
    "S5_Ridge": build_ridge_pipeline,
}
