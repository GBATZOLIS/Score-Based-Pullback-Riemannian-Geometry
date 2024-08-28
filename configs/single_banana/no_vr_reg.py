import ml_collections
import torch
import numpy as np
from datetime import timedelta

def get_config():
    config = ml_collections.ConfigDict()

    # Logging settings
    config.base_log_dir = "./results/single_banana"
    config.experiment = "lightning_no_vr_reg_U"
    config.eval_log_frequency = 100

    # Model settings
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
    config.premultiplication_by_U = True # new flag for premultiplication by U.T

    # Training settings
    config.epochs = 4000
    config.patience_epochs = 250
    config.checkpoint_frequency = 1
    config.loss = 'normalizing flow'
    config.std = 0.1 #the chosen std is critical and it depends on the dataset. We should create a rigorous method that estimates the optimal std.
    config.use_reg = True
    config.reg_factor = 1
    config.reg_type = 'isometry'
    config.use_cv = False

    # Data settings
    config.dataset_class = 'numpy_dataset'
    config.dataset = 'single_banana'
    config.data_path = "./data"
    config.d = 2
    config.batch_size = 64

    # Device settings
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization settings
    config.learning_rate = 0.0005

    # Optional loading of model checkpoints for resuming
    config.checkpoint = '/store/CIA/gb511/projects/riemannian-geometry/code/results/single_banana/lightning_no_vr_reg_U/checkpoints/last.ckpt'
    
    # Reproducibility
    config.seed = 1638128

    return config
