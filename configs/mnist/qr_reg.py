import ml_collections
import torch

def get_config():
    config = ml_collections.ConfigDict()

    # Logging settings
    config.base_log_dir = "./results/mnist"
    config.experiment = "qr_reg_sigma_25"
    config.eval_log_frequency = 10

    # Model settings
    config.diffeomorphism_class = 'image_diffeomorphism'
    config.actnorm = True
    config.alpha = 0.05
    config.coupling_layer_type = 'rational_quadratic_spline'
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
    config.dropout_prob = 0. #0.2

    # Training settings
    config.epochs = 10000
    config.checkpoint_frequency = 1
    config.std = 25 #the chosen std is critical and it depends on the dataset. We should create a rigorous method that estimates the optimal std.
    config.use_reg = True
    config.reg_factor = 0.001
    config.use_cv = False

    # Data settings
    config.dataset_class = 'mnist'
    config.dataset = 'mnist'
    config.digit = 1
    config.data_path = "./data"
    config.c = 1  # Number of channels
    config.h = 32 # Height
    config.w = 32 # Width
    config.d = config.c * config.h * config.w # Shape for the image data
    config.batch_size = 64

    # Device settings
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization settings
    config.learning_rate = 0.0002
    config.optimizer = 'AdamW'

    # Optional loading of model checkpoints for resuming
    config.load_phi_checkpoint = None
    config.load_psi_checkpoint = None

    # Reproducibility
    config.seed = 23

    return config
