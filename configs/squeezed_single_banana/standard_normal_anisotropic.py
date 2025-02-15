import ml_collections
import torch
import numpy as np
from datetime import timedelta

def get_config():
    config = ml_collections.ConfigDict()

    # Logging settings
    config.base_log_dir = "./results/squeezed_single_banana/affine"
    config.experiment = "standard_normal_anisotropic"
    config.eval_log_frequency = 100

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
    config.num_flow_steps = 2
    config.premultiplication_by_U = False # new flag for premultiplication by U.T

    # Training settings
    config.epochs = 1000
    config.patience_epochs = 100
    config.checkpoint_frequency = 1
    config.loss = 'normalizing flow'
    config.std = 0.074 #the chosen std is critical and it depends on the dataset. We should create a rigorous method that estimates the optimal std.
    config.use_reg = False
    config.reg_factor = 1
    config.reg_type = 'isometry'
    config.use_cv = False

    # Data settings
    config.dataset_class = 'numpy_dataset'
    config.dataset = 'squeezed_single_banana'
    config.data_path = "./data"
    config.d = 2
    config.batch_size = 64

    # Device settings
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization settings
    config.use_scheduler = True
    config.learning_rate = 0.0005

    # Optional loading of model checkpoints for resuming
    config.checkpoint = '/home/gb511/riemannian_geo/results/squeezed_single_banana/affine/standard_normal_anisotropic/checkpoints/checkpoint_epoch_259_loss_1.928.pth'
    
    # Reproducibility
    config.seed = 123

    return config
