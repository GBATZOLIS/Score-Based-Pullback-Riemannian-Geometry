import ml_collections
import torch
import numpy as np
from datetime import timedelta

def get_config():
    config = ml_collections.ConfigDict()

    # Logging settings
    config.base_log_dir = "./results/single_banana"
    config.experiment = "volume_preservation_affine"
    config.eval_log_frequency = 50

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
    config.num_flow_steps = 4

    # Training settings
    config.epochs = 1500 #max number of epochs. We use cosine learning rate annealing until config.epochs, where lr goes to 0.
    config.checkpoint_frequency = 1
    config.loss = 'loglikelihood' #choices:[denoising score matching, loglikelihood]
    config.std = 0.1
    config.use_reg = False
    config.reg_factor = 1.0
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
    config.load_phi_checkpoint = None
    config.load_psi_checkpoint = None

    # Reproducibility
    config.seed = 1638128

    return config
