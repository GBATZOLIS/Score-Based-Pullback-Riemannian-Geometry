import ml_collections
import torch

def get_config():
    config = ml_collections.ConfigDict()

    # Logging settings
    config.base_log_dir = "./results"
    config.experiment = "mnist"
    config.eval_log_frequency = 200

    # Model settings
    config.diffeomorphism_class = 'image_diffeomorphism'
    config.actnorm = True
    config.alpha = 0.05
    config.coupling_layer_type = "rational_quadratic_spline"
    config.hidden_channels = 96
    config.levels = 3
    config.multi_scale = False
    config.num_bits = 8
    config.num_res_blocks = 3
    config.preprocessing = "glow"
    config.resnet_batchnorm = True
    config.steps_per_level = 7
    config.spline_params = {
        "apply_unconditional_transform": False,
        "min_bin_height": 0.001,
        "min_bin_width": 0.001,
        "min_derivative": 0.001,
        "num_bins": 4,
        "tail_bound": 3.0
    }
    config.use_resnet = True
    config.dropout_prob = 0.2

    # Training settings
    config.epochs = 10000
    config.checkpoint_frequency = 1
    config.std = 1e-1
    config.use_reg = False
    config.reg_factor = 1.0

    # Data settings
    config.dataset_class = 'mnist'
    config.digit = 1
    config.data_path = "./data"
    config.c = 1  # Number of channels
    config.h = 32 # Height
    config.w = 32 # Width
    config.d = config.c * config.h * config.w # Shape for the image data
    config.batch_size = 32

    # Device settings
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization settings
    config.learning_rate = 0.0005

    # Optional loading of model checkpoints for resuming
    config.load_phi_checkpoint = None
    config.load_psi_checkpoint = None

    # Reproducibility
    config.seed = 23

    return config
