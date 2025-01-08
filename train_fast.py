import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from src.diffeomorphisms import get_diffeomorphism
from src.diffeomorphisms.utils import get_principal_components
from src.strongly_convex import get_strongly_convex
from src.training.train_utils import (
    EMA, load_config, load_model, save_model, resume_training,
    count_parameters, check_parameters_device, set_visible_gpus, set_seed
)
from src.training.optim_utils import get_optimizer_and_scheduler
from src.training.callbacks import check_manifold_properties
from src.training.loss import get_loss_function
from src.training.plot_utils import plot_data, plot_variances, log_diagonal_values
from src.training.data_utils import compute_mean_distance_and_sigma
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast  # AMP tools
from src.data import get_dataset
import torch._dynamo

# Suppress dynamo errors
torch._dynamo.config.suppress_errors = True

# Set which GPUs are visible
set_visible_gpus('1')

def main(config_path):
    config = load_config(config_path)

    # Set random seeds
    set_seed(config.seed)

    # Logging directories
    tensorboard_dir = os.path.join(config.base_log_dir, config.experiment, 'training_logs')
    checkpoint_dir = os.path.join(config.base_log_dir, config.experiment, 'checkpoints')
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # DataLoader setup
    dataset_class = get_dataset(config.dataset_class)
    train_dataset = dataset_class(config, split='train')
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    _, _, _, _ = compute_mean_distance_and_sigma(train_loader)
    plot_data(writer, train_loader, config.std, num_points=256)

    val_dataset = dataset_class(config, split='val')
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

    # Model initialization
    U, mean, stds = None, None, None
    if config.get('premultiplication_by_U', False):
        U, mean, stds = get_principal_components(train_loader, config.std)

    phi = torch.compile(get_diffeomorphism(config, U=U, mean=mean), mode='reduce-overhead')
    psi = torch.compile(get_strongly_convex(config, stds=stds), mode='reduce-overhead')

    phi_total_params, phi_trainable_params = count_parameters(phi)
    psi_total_params, psi_trainable_params = count_parameters(psi)
    print(f"Model Phi - Total Parameters: {phi_total_params}, Trainable Parameters: {phi_trainable_params}")
    print(f"Model Psi - Total Parameters: {psi_total_params}, Trainable Parameters: {psi_trainable_params}")

    # Optimizer and scheduler
    total_steps = config.epochs * len(train_loader)
    optimizer, scheduler = get_optimizer_and_scheduler(list(phi.parameters()) + list(psi.parameters()), config, total_steps)

    # Device configuration
    device = torch.device(config.device)
    phi = phi.to(device)
    psi = psi.to(device)

    ema_phi = EMA(model=phi, decay=0.999)
    ema_psi = EMA(model=psi, decay=0.999)

    check_parameters_device(phi)

    start_epoch, step, best_checkpoints, best_val_loss, epochs_no_improve, optimizer, scheduler = resume_training(
        config, phi, ema_phi, psi, ema_psi, load_model, get_optimizer_and_scheduler, total_steps, train_loader
    )

    if start_epoch == 0:
        if config.get('premultiplication_by_U', False):
            plot_variances(writer, stds, start_epoch)
        else:
            log_diagonal_values(psi, writer, start_epoch)

    loss_fn = get_loss_function(config)
    scaler = GradScaler()

    ema_phi.apply_shadow()
    ema_psi.apply_shadow()
    phi.eval()
    psi.eval()
    check_manifold_properties(config.dataset, phi, psi, writer, 0, device, val_loader, config.d)
    ema_phi.restore()
    ema_psi.restore()

    warm_up_epochs = 5

    for epoch in range(start_epoch, config.epochs):
        phi.train()
        psi.train()

        enable_amp = epoch >= warm_up_epochs

        train_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{config.epochs}", leave=False)
        total_loss = 0
        total_density_learning_loss = 0
        total_reg_loss = 0
        total_iso_reg_loss = 0
        total_volume_reg_loss = 0
        total_hessian_reg_loss = 0

        for data in train_iterator:
            x = data[0].to(device) if isinstance(data, list) else data.to(device)

            optimizer.zero_grad()

            if enable_amp:
                with autocast():
                    loss, density_learning_loss, reg_loss, iso_reg, volume_reg, hessian_reg = loss_fn(phi, psi, x, train=True, device=device)
            else:
                loss, density_learning_loss, reg_loss, iso_reg, volume_reg, hessian_reg = loss_fn(phi, psi, x, train=True, device=device)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(phi.parameters()) + list(psi.parameters()), max_norm=1)
            scaler.step(optimizer)
            scaler.update()

            # After updating the optimizer, also update the EMA models
            ema_phi.update()
            ema_psi.update()

            if scheduler:
                scheduler.step()

            # Log the losses for TensorBoard
            writer.add_scalar("Loss/Train Step", loss.item(), step)
            writer.add_scalar(f"{loss_fn.loss_name} Train Step", density_learning_loss.item(), step)
            writer.add_scalar("Loss/Regularization Train Step", reg_loss.item(), step)
            writer.add_scalar("Loss/Isometry Regularization Train Step", iso_reg.item(), step)
            writer.add_scalar("Loss/Volume Regularization Train Step", volume_reg.item(), step)
            writer.add_scalar("Loss/Hessian Regularization Train Step", hessian_reg.item(), step)

            if step % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("Learning Rate", current_lr, step)

            step += 1

            total_loss += loss.item()
            total_density_learning_loss += density_learning_loss.item()
            total_reg_loss += reg_loss.item()
            total_iso_reg_loss += iso_reg.item()
            total_volume_reg_loss += volume_reg.item()
            total_hessian_reg_loss += hessian_reg.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_density_learning_loss = total_density_learning_loss / len(train_loader)
        avg_train_reg_loss = total_reg_loss / len(train_loader)
        avg_train_iso_reg_loss = total_iso_reg_loss / len(train_loader)
        avg_train_volume_reg_loss = total_volume_reg_loss / len(train_loader)
        avg_train_hessian_reg_loss = total_hessian_reg_loss / len(train_loader)

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar(f"{loss_fn.loss_name} Train", avg_train_density_learning_loss, epoch)
        writer.add_scalar("Loss/Regularization Train", avg_train_reg_loss, epoch)
        writer.add_scalar("Loss/Isometry Regularization Train", avg_train_iso_reg_loss, epoch)
        writer.add_scalar("Loss/Volume Regularization Train", avg_train_volume_reg_loss, epoch)
        writer.add_scalar("Loss/Hessian Regularization Train", avg_train_hessian_reg_loss, epoch)

        ema_phi.apply_shadow()
        ema_psi.apply_shadow()

        phi.eval()
        psi.eval()

        # Validation Loop
        val_iterator = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{config.epochs}", leave=False)
        total_val_loss = 0
        total_val_density_learning_loss = 0
        total_val_reg_loss = 0
        total_val_iso_reg_loss = 0
        total_val_volume_reg_loss = 0
        total_val_hessian_reg_loss = 0

        with torch.no_grad():  # Disable gradient computation
            for data in val_iterator:
                # Move data to the specified device (CPU/GPU)
                x = data[0].to(device) if isinstance(data, list) else data.to(device)

                # No mixed-precision `autocast` is used here
                val_loss, val_density_learning_loss, val_reg_loss, val_iso_reg, val_volume_reg, val_hessian_reg = loss_fn(
                    phi, psi, x, train=False, device=device
                )

                # Accumulate all validation metrics
                total_val_loss += val_loss.item()
                total_val_density_learning_loss += val_density_learning_loss.item()
                total_val_reg_loss += val_reg_loss.item()
                total_val_iso_reg_loss += val_iso_reg.item()
                total_val_volume_reg_loss += val_volume_reg.item()
                total_val_hessian_reg_loss += val_hessian_reg.item()

        # Compute averages of all validation metrics
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_density_learning_loss = total_val_density_learning_loss / len(val_loader)
        avg_val_reg_loss = total_val_reg_loss / len(val_loader)
        avg_val_iso_reg_loss = total_val_iso_reg_loss / len(val_loader)
        avg_val_volume_reg_loss = total_val_volume_reg_loss / len(val_loader)
        avg_val_hessian_reg_loss = total_val_hessian_reg_loss / len(val_loader)

        # Log validation metrics to TensorBoard
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar(f"{loss_fn.loss_name} Validation", avg_val_density_learning_loss, epoch)
        writer.add_scalar("Loss/Regularization Validation", avg_val_reg_loss, epoch)
        writer.add_scalar("Loss/Isometry Regularization Validation", avg_val_iso_reg_loss, epoch)
        writer.add_scalar("Loss/Volume Regularization Validation", avg_val_volume_reg_loss, epoch)
        writer.add_scalar("Loss/Hessian Regularization Validation", avg_val_hessian_reg_loss, epoch)

        # Print summary of validation results
        print(f"Epoch {epoch+1}/{config.epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        if (epoch + 1) % config.eval_log_frequency == 0:
            check_manifold_properties(config.dataset, phi, psi, writer, epoch, device, val_loader, config.d)

        ema_phi.restore()
        ema_psi.restore()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.get('patience_epochs', 100):
            print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
            break

        if epoch % config.checkpoint_frequency == 0:
            save_model(phi, ema_phi, psi, ema_psi, epoch, avg_val_loss, checkpoint_dir, best_checkpoints, step, best_val_loss, epochs_no_improve, optimizer, scheduler)

    writer.close()
    print("Training completed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training with Config File")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    main(args.config)
