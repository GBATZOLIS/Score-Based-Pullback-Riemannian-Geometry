import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from src.diffeomorphisms import get_diffeomorphism
from src.strongly_convex import get_strongly_convex
from src.training.train_utils import EMA, load_config, get_full_checkpoint_path, set_visible_gpus, set_seed, resume_training, load_model
from src.training.callbacks import check_manifold_properties
from src.data import get_dataset
from src.training.optim_utils import get_optimizer_and_scheduler
from src.diffeomorphisms.utils import get_principal_components  # Import the PCA computation function

# Set which GPUs are visible
set_visible_gpus('3')

def main(config_path):
    config = load_config(config_path)

    # Set random seeds
    set_seed(config.seed)

    # Handle logging directories
    checkpoint_dir = os.path.join(config.base_log_dir, config.experiment, 'checkpoints')
    tensorboard_dir = os.path.join(config.base_log_dir, config.experiment, 'eval_logs')
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # DataLoader for validation data
    dataset_class = get_dataset(config.dataset_class)
    train_dataset = dataset_class(config, split='train')
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataset = dataset_class(config, split='val')
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize PCA rotation matrix and mean if the flag is enabled
    U, mean = None, None
    if config.get('premultiplication_by_U', False):
        U, mean = get_principal_components(train_loader)
        print(f'U.size(): {U.size()}')
        print(f'mean: {mean}')

    # Initialize the models with PCA rotation and mean if applicable
    phi = get_diffeomorphism(config, U=U, mean=mean)  # Pass the PCA matrix and mean if applicable
    psi = get_strongly_convex(config)

    # Device configuration
    device = torch.device(config.device)
    phi = phi.to(device)
    psi = psi.to(device)

    # Initialize EMA handlers
    ema_phi = EMA(model=phi, decay=0.999)
    ema_psi = EMA(model=psi, decay=0.999)

    # Resume training from checkpoint
    start_epoch, step, best_checkpoints, best_val_loss, epochs_no_improve, optimizer, scheduler = resume_training(
        config, phi, ema_phi, psi, ema_psi, load_model, get_optimizer_and_scheduler, 0, val_loader
    )

    # Apply EMA shadow variables before evaluation
    ema_phi.apply_shadow()
    ema_psi.apply_shadow()
    phi.eval()
    psi.eval()

    # Evaluate and log manifold properties
    epoch = start_epoch - 1  # epoch starts from 0 in training
    check_manifold_properties(config.dataset, phi, psi, writer, epoch, device, val_loader, config.d, True)

    # Restore original parameters after evaluation
    ema_phi.restore()
    ema_psi.restore()

    writer.close()
    print("Evaluation completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluation with Config File")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    main(args.config)
