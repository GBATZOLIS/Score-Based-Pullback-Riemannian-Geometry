import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from src.diffeomorphisms import get_diffeomorphism
from src.strongly_convex.learnable_psi import LearnablePsi
from src.training.train_utils import EMA, load_model, save_model, get_log_density_fn, get_score_fn, WarmUpScheduler, count_parameters
from src.training.callbacks import check_manifold_properties
from tqdm import tqdm
from torch.autograd.functional import jvp
import importlib.util
import sys
from src.data import get_dataset


def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config_module"] = config_module
    spec.loader.exec_module(config_module)
    return config_module.get_config()


def compute_loss(phi, psi, x, args_std, train=True, use_reg=False, reg_factor=1, device='cuda:0'):
    def score_matching_loss():
        std = args_std * torch.ones_like(x)
        score_fn = get_score_fn(phi, psi, train=train)
        z = torch.randn_like(x)
        x_pert = x + std * z
        score = score_fn(x_pert)
        target = -1 * z
        loss = torch.mean((std * score - target)**2)
        return loss

    def regularization_term():
        batch_size, dim = x.shape
        v = torch.randn(batch_size, dim, device=device)
        v /= v.norm(dim=1, keepdim=True)

        if not train:
            torch.set_grad_enabled(True)

        _, Jv = jvp(phi, (x,), (v,))

        if not train:
            torch.set_grad_enabled(False)
        
        norms = Jv.norm(dim=1)
        return torch.mean((norms - 1) ** 2)
    
    if use_reg:
        loss = score_matching_loss() + reg_factor * regularization_term()
    else:
        loss = score_matching_loss()
    
    return loss

def main(config_path):
    config = load_config(config_path)

    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Handle logging directories
    tensorboard_dir = os.path.join(config.base_log_dir, config.experiment, 'training_logs')
    checkpoint_dir = os.path.join(config.base_log_dir, config.experiment, 'checkpoints')

    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Initialize the models
    phi = get_diffeomorphism(config.diffeomorphism_class)(config)
    psi = LearnablePsi(config.d)

    # Print model summaries
    phi_total_params, phi_trainable_params = count_parameters(phi)
    psi_total_params, psi_trainable_params = count_parameters(psi)
    print(f"Model Phi - Total Parameters: {phi_total_params}, Trainable Parameters: {phi_trainable_params}")
    print(f"Model Psi - Total Parameters: {psi_total_params}, Trainable Parameters: {psi_trainable_params}")

    # Initialize EMA handlers
    ema_phi = EMA(model=phi, decay=0.999)
    ema_psi = EMA(model=psi, decay=0.999)

    # DataLoader for synthetic training and validation data
    dataset_class = get_dataset(config.dataset_class)
    train_dataset = dataset_class(config, split='train')
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    val_dataset = dataset_class(config, split='val')
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Optimizer and scheduler
    optimizer = optim.Adam(list(phi.parameters()) + list(psi.parameters()), lr=0)
    scheduler = WarmUpScheduler(optimizer, config.learning_rate, warmup_steps=2000)

    # Device configuration
    device = torch.device(config.device)
    phi.to(device)
    psi.to(device)

    # Load checkpoints if specified
    if config.load_phi_checkpoint:
        epoch_phi, loss_phi = load_model(phi, None, config.load_phi_checkpoint, "Phi")
        ema_path_phi = config.load_phi_checkpoint.replace('.pth', '_EMA.pth')
        load_model(phi, ema_phi, ema_path_phi, "Phi EMA", is_ema=True)

    if config.load_psi_checkpoint:
        epoch_psi, loss_psi = load_model(psi, None, config.load_psi_checkpoint, "Psi")
        ema_path_psi = config.load_psi_checkpoint.replace('.pth', '_EMA.pth')
        load_model(psi, ema_psi, ema_path_psi, "Psi EMA", is_ema=True)

    step = 0
    best_checkpoints = []

    # Training and Validation loop
    for epoch in range(config.epochs):
        phi.train()
        psi.train()

        train_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{config.epochs}", leave=False)
        total_loss = 0
        for data in train_iterator:
            if isinstance(data, list):
                x = data[0]
            x = x.to(device)
            
            loss = compute_loss(phi, psi, x, config.std, train=True, use_reg=config.use_reg, reg_factor=config.reg_factor, device=device)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(list(phi.parameters()) + list(psi.parameters()), max_norm=1)
            optimizer.step()

            ema_phi.update()
            ema_psi.update()

            scheduler.step()

            if step % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("Learning Rate", current_lr, step)

            step += 1
            train_iterator.set_postfix(loss=loss.item())
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        
        # Validation
        ema_phi.apply_shadow()
        ema_psi.apply_shadow()

        phi.eval()
        psi.eval()
        val_iterator = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{config.epochs}", leave=False)
        total_val_loss = 0
        with torch.no_grad():
            for data in val_iterator:
                if isinstance(data, list):
                    x = data[0]
                x = x.to(device)

                val_loss = compute_loss(phi, psi, x, config.std, train=False, use_reg=config.use_reg, reg_factor=config.reg_factor, device=device)

                total_val_loss += val_loss.item()
                val_iterator.set_postfix(val_loss=val_loss.item())

        avg_val_loss = total_val_loss / len(val_loader)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)

        print(f"Epoch {epoch+1}/{config.epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        if (epoch + 1) % config.eval_log_frequency == 0:
            check_manifold_properties(phi, psi, writer, epoch)
        
        ema_phi.restore()
        ema_psi.restore()
        
        if (epoch + 1) % config.checkpoint_frequency == 0:
            save_model(phi, ema_phi, epoch, avg_train_loss, "Phi", checkpoint_dir, best_checkpoints)
            save_model(psi, ema_psi, epoch, avg_train_loss, "Psi", checkpoint_dir, best_checkpoints)

    writer.close()
    print("Training completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training with Config File")
    parser.add_argument("--config-path", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    main(args.config_path)
