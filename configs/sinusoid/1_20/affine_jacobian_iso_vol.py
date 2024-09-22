import ml_collections
import torch
import numpy as np
from datetime import timedelta

def get_config():
    config = ml_collections.ConfigDict()

    # Logging settings
    config.base_log_dir = "./results/sinusoid/1_20"
    config.experiment = "affine_jacobian_iso_vol_weights_1_1"
    config.eval_log_frequency = 50

    # Model settings
    ## Strongly convex function settings
    config.strongly_convex_class = 'learnable_psi'
    
    ## Diffeomorphism settings
    config.diffeomorphism_class = 'euclidean_diffeomorphism'
    config.base_transform_type = 'affine'
    config.hidden_features = 64
    config.num_transform_blocks = 2
    config.use_batch_norm = 0
    config.dropout_probability = 0.0
    config.num_bins = 128
    config.apply_unconditional_transform = 0
    config.min_bin_width = 1e-3
    config.num_flow_steps = 8
    config.premultiplication_by_U = False # new flag for premultiplication by U.T

    # Training settings
    config.epochs = 2000
    config.patience_epochs = 200
    config.checkpoint_frequency = 1
    config.loss = 'normalizing flow'
    config.std = 0. #the chosen std is critical and it depends on the dataset. We should create a rigorous method that estimates the optimal std.
    config.use_reg = True
    config.reg_factor = 1
    config.lambda_iso = 1
    config.lambda_vol = 1
    config.lambda_hessian = 1
    config.reg_type = 'isometry+volume'
    config.reg_iso_type = 'orthogonal-jacobian'
    config.use_cv = False

    # Data settings
    config.dataset_class = 'numpy_dataset'
    config.dataset = 'sinusoid_1_20'
    config.data_path = "./data"
    config.d = 20
    config.batch_size = 64
    config.data_range = [-8, 8]
    
    # Device settings
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization settings
    config.use_scheduler = True
    config.learning_rate = 3e-4

    # Optional loading of model checkpoints for resuming
    config.checkpoint = None
    
    # Reproducibility
    config.seed = 12

    return config
