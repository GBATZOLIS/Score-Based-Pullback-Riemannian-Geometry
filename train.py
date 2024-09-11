import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import os
from src.diffeomorphisms import get_diffeomorphism
from src.diffeomorphisms.utils import get_principal_components  # Import the PCA computation function
from src.strongly_convex import get_strongly_convex
from src.training.train_utils import EMA, load_config, load_model, save_model, resume_training, get_log_density_fn, get_score_fn, count_parameters, check_parameters_device, set_visible_gpus, set_seed, get_full_checkpoint_path
from src.training.optim_utils import get_optimizer_and_scheduler
from src.training.callbacks import check_manifold_properties, check_manifold_properties_images
from src.training.loss import get_loss_function
from src.training.plot_utils import plot_data, plot_variances
from src.training.data_utils import compute_mean_distance_and_sigma
from tqdm import tqdm
from torch.autograd.functional import jvp
from src.data import get_dataset

# Set which GPUs are visible
set_visible_gpus('3')

def main(config_path):
    config = load_config(config_path)

    # Set random seeds
    set_seed(config.seed)

    # Handle logging directories
    tensorboard_dir = os.path.join(config.base_log_dir, config.experiment, 'training_logs')
    checkpoint_dir = os.path.join(config.base_log_dir, config.experiment, 'checkpoints')

    writer = SummaryWriter(log_dir=tensorboard_dir)

    # DataLoader for synthetic training and validation data
    dataset_class = get_dataset(config.dataset_class)
    train_dataset = dataset_class(config, split='train')
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    _, _, _, _ = compute_mean_distance_and_sigma(train_loader)
    plot_data(writer, train_loader, config.std, num_points=256)

    val_dataset = dataset_class(config, split='val')
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize the models
    # Compute PCA rotation matrix and mean if the flag is enabled
    U, mean, stds = None, None, None
    if config.get('premultiplication_by_U', False):
        U, mean, stds = get_principal_components(train_loader, config.std)
        print(f'U.size(): {U.size()}')
        print(f'mean: {mean}')
        print(f'stds: {stds[:30]}')

    phi = get_diffeomorphism(config, U=U, mean=mean)  # Pass the PCA matrix and mean if applicable
    psi = get_strongly_convex(config, stds=stds)

    ## Print model summaries
    phi_total_params, phi_trainable_params = count_parameters(phi)
    psi_total_params, psi_trainable_params = count_parameters(psi)
    print(f"Model Phi - Total Parameters: {phi_total_params}, Trainable Parameters: {phi_trainable_params}")
    print(f"Model Psi - Total Parameters: {psi_total_params}, Trainable Parameters: {psi_trainable_params}")

    # Calculate total steps for scheduler
    total_steps = config.epochs * len(train_loader)

    # Optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(list(phi.parameters()) + list(psi.parameters()), config, total_steps)

    # Device configuration
    device = torch.device(config.device)
    phi = phi.to(device)
    psi = psi.to(device)

    # Initialize EMA handlers
    ema_phi = EMA(model=phi, decay=0.999)
    ema_psi = EMA(model=psi, decay=0.999)

    # Print the device of each parameter in phi
    check_parameters_device(phi)
        
    start_epoch, step, best_checkpoints, best_val_loss, epochs_no_improve, optimizer, scheduler = resume_training(
        config, phi, ema_phi, psi, ema_psi, load_model, get_optimizer_and_scheduler, total_steps, train_loader
    )

    if start_epoch == 0 and config.get('premultiplication_by_U', False):
        plot_variances(writer, stds, start_epoch)

    # Get the appropriate loss function
    loss_fn = get_loss_function(config)

    # Training and Validation loop
    for epoch in range(start_epoch, config.epochs):
        phi.train()
        psi.train()

        train_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{config.epochs}", leave=False)
        total_loss = 0
        total_density_learning_loss = 0
        total_reg_loss = 0
        for data in train_iterator:
            if isinstance(data, list):
                x = data[0]
            else:
                x = data

            x = x.to(device)
            
            loss, density_learning_loss, reg_loss = loss_fn(phi, psi, x, train=True, device=device)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(list(phi.parameters()) + list(psi.parameters()), max_norm=1)
            optimizer.step()

            ema_phi.update()
            ema_psi.update()

            if scheduler:  # Step the scheduler if it's being used
                scheduler.step()

            writer.add_scalar("Loss/Train Step", loss.item(), step)
            writer.add_scalar(f"{loss_fn.loss_name} Train Step", density_learning_loss.item(), step)
            writer.add_scalar("Loss/Regularization Train Step", reg_loss.item(), step)

            if step % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("Learning Rate", current_lr, step)

            step += 1
            train_iterator.set_postfix(loss=loss.item())
            total_loss += loss.item()
            total_density_learning_loss += density_learning_loss.item()
            if reg_loss is not None:
                total_reg_loss += reg_loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_density_learning_loss = total_density_learning_loss / len(train_loader)
        avg_train_reg_loss = total_reg_loss / len(train_loader) if total_reg_loss > 0 else 0

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar(f"{loss_fn.loss_name} Train", avg_train_density_learning_loss, epoch)
        writer.add_scalar("Loss/Regularization Train", avg_train_reg_loss, epoch)
        
        # Validation
        ema_phi.apply_shadow()
        ema_psi.apply_shadow()

        phi.eval()
        psi.eval()
        val_iterator = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{config.epochs}", leave=False)
        total_val_loss = 0
        total_val_density_learning_loss = 0
        total_val_reg_loss = 0
        with torch.no_grad():
            for data in val_iterator:
                if isinstance(data, list):
                    x = data[0]
                else:
                    x = data

                x = x.to(device)

                val_loss, val_density_learning_loss, val_reg_loss = loss_fn(phi, psi, x, train=False, device=device)

                total_val_loss += val_loss.item()
                total_val_density_learning_loss += val_density_learning_loss.item()
                if val_reg_loss is not None:
                    total_val_reg_loss += val_reg_loss.item()
                val_iterator.set_postfix(val_loss=val_loss.item())

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_density_learning_loss = total_val_density_learning_loss / len(val_loader)
        avg_val_reg_loss = total_val_reg_loss / len(val_loader) if total_val_reg_loss > 0 else 0
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar(f"{loss_fn.loss_name} Validation", avg_val_density_learning_loss, epoch)
        writer.add_scalar("Loss/Regularization Validation", avg_val_reg_loss, epoch)

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
            print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss for {config.patience_epochs} consecutive epochs.")
            break
        
        # Checkpoint saving
        if (epoch + 1) % config.checkpoint_frequency == 0:
            save_model(phi, ema_phi, psi, ema_psi, epoch, avg_val_loss, 
                       checkpoint_dir, best_checkpoints, step, 
                       best_val_loss, epochs_no_improve, optimizer, scheduler)

    writer.close()
    print("Training completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training with Config File")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    main(args.config)