import ml_collections
import torch

#THIS CONFIG WORKS!
def get_config():
    config = ml_collections.ConfigDict()

    # Logging settings
    config.base_log_dir = "./results/blobs"
    config.experiment = "our_method_softplus"
    config.eval_log_frequency = 10

    # Psi - strongly convex settings
    config.strongly_convex_class = 'learnable_psi'
    config.use_softplus = True

    # Phi Model settings
    config.diffeomorphism_class = 'image_diffeomorphism'
    config.actnorm = True
    config.alpha = 0.05
    config.coupling_layer_type = 'rational_quadratic_spline'
    config.hidden_channels = 96
    config.levels = 3
    config.multi_scale = False
    config.num_bits = 8
    config.num_res_blocks = 3
    config.preprocessing = None

    config.use_resnet = True
    config.resnet_batchnorm = False

    config.steps_per_level = 7
    config.spline_params = {
        "apply_unconditional_transform": False,
        "min_bin_height": 0.001,
        "min_bin_width": 0.001,
        "min_derivative": 0.001,
        "num_bins": 11,
        "tail_bound": 10
    }
    config.dropout_prob = 0. #0.2
    config.premultiplication_by_U = False # new flag for premultiplication by U.T

    # Training settings
    config.epochs = 5000
    config.patience_epochs = 1000
    config.checkpoint_frequency = 1
    config.loss = 'normalizing flow'
    config.std = 0.05 #the chosen std is critical and it depends on the dataset. We should create a rigorous method that estimates the optimal std.
    config.use_reg = True
    config.reg_factor = 100
    config.reg_type = 'isometry'
    config.use_cv = False

    # Updated Data settings
    config.dataset_class = 'fixed_gaussians_manifold'  # Matches the class name
    config.dataset = 'blobs'
    config.data_path = "./data"  # Placeholder for data path (not used in synthetic data)
    config.num_gaussians = 10  # Number of Gaussians per image
    config.std_range = [1, 5]  # Standard deviation range for Gaussians
    config.data_samples = 50000 # Number of samples in the dataset
    config.image_size = 32  # Image dimensions (32x32)
    config.split = [0.8, 0.1, 0.1]  # Train/val/test split ratios
    config.c = 1  # Number of channels (grayscale)
    config.h = 32  # Height of the images
    config.w = 32  # Width of the images
    config.d = config.c * config.h * config.w  # Flattened dimension of the image
    config.batch_size = 64  # Batch size for training

    # Device settings
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization settings
    config.learning_rate = 0.0002
    config.optimizer = 'AdamW'

    # Optional loading of model checkpoints for resuming
    config.checkpoint = None

    # Reproducibility
    config.seed = 23

    return config