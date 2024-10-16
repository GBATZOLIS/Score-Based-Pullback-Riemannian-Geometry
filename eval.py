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
from src.evaluation import get_ground_truth_pullback_manifold, get_learned_pullback_manifold, geodesic_error, geodesic_variation_error, evaluate_manifold_maps, rae_evaluation, rae_projection
import matplotlib.pyplot as plt
import numpy as np

# Set which GPUs are visible
set_visible_gpus('2')

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
    test_dataset = dataset_class(config, split='test')
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize PCA rotation matrix and mean if the flag is enabled
    U, mean, stds = None, None, None
    if config.get('premultiplication_by_U', False):
        U, mean, stds = get_principal_components(train_loader)
        print(f'U.size(): {U.size()}')
        print(f'mean: {mean}')
        print(f'stds: {stds}')

    # Initialize the models with PCA rotation and mean if applicable
    phi = get_diffeomorphism(config, U=U, mean=mean)  # Pass the PCA matrix and mean if applicable
    psi = get_strongly_convex(config, stds=stds)

    # Device configuration
    device = torch.device(config.device)
    #print(device)
    phi = phi.to(device)
    psi = psi.to(device)

    # Initialize EMA handlers
    ema_phi = EMA(model=phi, decay=0.999)
    ema_psi = EMA(model=psi, decay=0.999)

    # Resume training from checkpoint
    start_epoch, step, best_checkpoints, best_val_loss, epochs_no_improve, optimizer, scheduler = resume_training(
        config, phi, ema_phi, psi, ema_psi, load_model, get_optimizer_and_scheduler, 0, val_loader
    )


    epoch = start_epoch - 1  # epoch starts from 0 in training

    # Apply EMA shadow variables before evaluation
    ema_phi.apply_shadow()
    ema_psi.apply_shadow()
    phi.eval()
    psi.eval()
    phi = phi.to(device)
    psi = psi.to(device)
    
    rae_projection(psi, phi, tensorboard_dir, device, train_loader)
    #rae_evaluation(psi, phi, test_loader, tensorboard_dir, device)

    #compute geodesic error and the geodesic variation error
    #learned_pullback_manifold = get_learned_pullback_manifold(phi, psi)
    #ground_truth_pullback_manifold = get_ground_truth_pullback_manifold(config)

    # Evaluate manifold maps and plot exponential maps and geodesics
    #evaluate_manifold_maps(learned_pullback_manifold, ground_truth_pullback_manifold, test_loader, device, writer, epoch)

    #geodesic_error_mean, geodesic_error_std = geodesic_error(learned_pullback_manifold, ground_truth_pullback_manifold, test_loader, device)
    #variation_error_mean, variation_error_std = geodesic_variation_error(learned_pullback_manifold, ground_truth_pullback_manifold, test_loader, device)

    #print(f'geodesic error (mean/std):({geodesic_error_mean:.4f}/{geodesic_error_std:.4f})')
    #print(f'variation error (mean/std):({variation_error_mean:.4f}/{variation_error_std:.4f})')

    # Evaluate and log manifold properties
    #check_manifold_properties(config.dataset, phi, psi, writer, epoch, device, val_loader, config.d, True)
    


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
