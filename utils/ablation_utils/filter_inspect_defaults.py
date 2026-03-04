"""Default configs for filter-learning inspection experiment."""

DEFAULT_FILTER_INSPECT_TEST_CONFIG = {
    "airport": [
        ("USA", "BRAZIL"),
    ],
}

DEFAULT_FILTER_INSPECT_EXPT_CONFIG = {
    "seed": 0,
    "device": "cuda:6",
    "out_dir": "../../__saved__/analysis/filter_inspect",
    "max_layers": 5,
    "metric_sample_size": 2000,
    "kernel_mul": 2.0,
    "kernel_num": 5,
    "fix_sigma": None,
    "cheb_k": 5,
    "cheb_lambda_max": 2.0,
    "train_epochs": 200,
    "train_lr": 1e-2,
    "train_weight_decay": 0.0,
    "train_log_interval": 10,
    "temp_reg": 1e-4,
    "edge_reg": 1e-4,
    "edge_eps": 1e-6,
    "struct_loss_mode": "adjacency",  # "adjacency" or "ppmi"
    "struct_bpr_weight": 0.0,
    "struct_bpr_samples": 2048,
    "struct_bpr_margin": 0.0,
    "ppmi_path_len": 5,
    "ppmi_pos_threshold": 0.0,
    "target_poly_coeffs": [0.0, 0.2, 0.5, 0.2],
    "target_max_power": 3,
    "coeff_plot_max_order": 8,
}
