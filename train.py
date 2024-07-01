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
from src.training.plot_utils import plot_data
from tqdm import tqdm
from torch.autograd.functional import jvp
import importlib.util
import sys
from src.data import get_dataset

from collections import defaultdict

def set_visible_gpus(gpus):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

# Set which GPUs are visible
set_visible_gpus('1')


def check_parameters_device(model):
    device_count = defaultdict(int)

    for param in model.parameters():
        device_count[param.device] += 1

    if len(device_count) == 1:
        device = next(iter(device_count))
        print(f"All parameters are on the same device: {device}")
    else:
        print("Parameters are on different devices:")
        for device, count in device_count.items():
            print(f"{count} parameters are on device: {device}")


def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config_module"] = config_module
    spec.loader.exec_module(config_module)
    return config_module.get_config()

class EMAForScalar:
    def __init__(self, decay=0.999):
        self.decay = decay
        self.shadow = None

    def update(self, value):
        if self.shadow is None:
            self.shadow = value
        else:
            self.shadow = self.decay * self.shadow + (1 - self.decay) * value

    def get(self):
        return self.shadow

def compute_loss_variance_reduction(phi, psi, x, args_std, train=True, use_cv=False, use_reg=False, reg_factor=1, device='cuda:0'):
    def score_matching_loss():
        x.requires_grad_(True)  # Ensure x requires gradients
        batch_size = x.size(0)
        std = args_std * torch.ones_like(x, device=device)
        score_fn = get_score_fn(phi, psi, train=train)
        z = torch.randn_like(x, device=device)
        x_pert = x + std * z
        score = score_fn(x_pert)
        
        term_inside_norm = (z / std + score)
        loss = torch.mean(torch.norm(term_inside_norm.view(batch_size, -1), dim=1)**2) / 2

        if use_cv:
            # Calculate the control variate c_theta(x, z)
            grad_log_p = score_fn(x)  # ∇_x log p_theta(x)
            control_variate = (2 / args_std) * (z * grad_log_p).view(batch_size, -1).sum(dim=1) + (z.view(batch_size, -1).norm(dim=1) ** 2 / args_std ** 2) - torch.prod(torch.tensor(x.shape[1:], device=device)) / args_std ** 2
            loss -= torch.mean(control_variate)

        return loss

    def regularization_term():
        #x.requires_grad_(True)  # Ensure x requires gradients
        batch_size, *dims = x.shape
        d = torch.prod(torch.tensor(dims, device=device))
        v = torch.randn(batch_size, *dims, device=device, requires_grad=True)
        v_norm = v.view(batch_size, -1).norm(dim=1, keepdim=True).view(batch_size, *[1]*len(dims))
        v = v / v_norm

        if not train:
            torch.set_grad_enabled(True)

        _, Jv = torch.autograd.functional.jvp(lambda x: phi(x), (x,), (v,), create_graph=True)

        if not train:
            torch.set_grad_enabled(False)
        
        norms = Jv.view(batch_size, -1).norm(dim=1)
        return torch.mean((norms - 1) ** 2)



    score_loss = score_matching_loss()
    reg_loss = regularization_term()

    if use_reg:
        loss = score_loss + reg_factor * reg_loss
    else:
        loss = score_loss
    
    return loss, score_loss, reg_loss

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

    # DataLoader for synthetic training and validation data
    dataset_class = get_dataset(config.dataset_class)
    train_dataset = dataset_class(config, split='train')
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    plot_data(writer, train_loader, num_points=256)

    val_dataset = dataset_class(config, split='val')
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Optimizer and scheduler
    optimizer = optim.Adam(list(phi.parameters()) + list(psi.parameters()), lr=0)
    scheduler = WarmUpScheduler(optimizer, config.learning_rate, warmup_steps=2000)

    # Device configuration
    device = torch.device(config.device)
    phi = phi.to(device)
    psi = psi.to(device)

    # Initialize EMA handlers
    ema_phi = EMA(model=phi, decay=0.999)
    ema_psi = EMA(model=psi, decay=0.999)
    '''ema_normalized_reg_factor = EMAForScalar(decay=0.999)'''

    # Print the device of each parameter in phi
    check_parameters_device(phi)
        
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
        total_score_loss = 0
        total_reg_loss = 0
        for data in train_iterator:
            if isinstance(data, list):
                x = data[0]
            else:
                x = data

            x = x.to(device)
            
            loss, score_loss, reg_loss = compute_loss_variance_reduction(phi, psi, x, config.std, train=True, 
                                                               use_cv=config.use_cv, use_reg=config.use_reg, reg_factor=config.reg_factor, device=device)
            '''
            if config.use_reg:
                # Compute gradients for reg_loss
                optimizer.zero_grad()
                reg_loss.backward(retain_graph=True)
                
                # Compute gradients norm for both phi and psi parameters
                reg_grad_norm_phi = torch.norm(torch.stack([p.grad.norm() for p in phi.parameters() if p.grad is not None]))
                reg_grad_norm = torch.norm(torch.tensor([reg_grad_norm_phi]))

                # Compute gradients for score_loss
                optimizer.zero_grad()
                score_loss.backward(retain_graph=True)
                
                # Compute gradients norm for both phi and psi parameters
                score_grad_norm_phi = torch.norm(torch.stack([p.grad.norm() for p in phi.parameters() if p.grad is not None]))
                score_grad_norm_psi = torch.norm(torch.stack([p.grad.norm() for p in psi.parameters() if p.grad is not None]))
                score_grad_norm = torch.norm(torch.tensor([score_grad_norm_phi]))

                # Normalize reg_factor
                normalized_reg_factor = score_grad_norm / reg_grad_norm
                ema_normalized_reg_factor.update(normalized_reg_factor.item())
            '''


            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(list(phi.parameters()) + list(psi.parameters()), max_norm=1)
            optimizer.step()

            ema_phi.update()
            ema_psi.update()

            scheduler.step()

            writer.add_scalar("Loss/Train Step", loss.item(), step)
            writer.add_scalar("Loss/Score Train Step", score_loss.item(), step)
            writer.add_scalar("Loss/Regularization Train Step", reg_loss.item(), step)
            
            '''
            if config.use_reg:
                writer.add_scalar("Gradients/score_grad_norm", score_grad_norm.item(), step)
                writer.add_scalar("Gradients/reg_grad_norm", reg_grad_norm.item(), step)
                writer.add_scalar("Gradients/normalized_reg_factor", ema_normalized_reg_factor.get(), step)
            '''

            if step % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("Learning Rate", current_lr, step)
                

            step += 1
            train_iterator.set_postfix(loss=loss.item())
            total_loss += loss.item()
            total_score_loss += score_loss.item()
            if reg_loss is not None:
                total_reg_loss += reg_loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_score_loss = total_score_loss / len(train_loader)
        avg_train_reg_loss = total_reg_loss / len(train_loader) if total_reg_loss > 0 else 0

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Score Train", avg_train_score_loss, epoch)
        writer.add_scalar("Loss/Regularization Train", avg_train_reg_loss, epoch)
        
        # Validation
        ema_phi.apply_shadow()
        ema_psi.apply_shadow()

        phi.eval()
        psi.eval()
        val_iterator = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{config.epochs}", leave=False)
        total_val_loss = 0
        total_val_score_loss = 0
        total_val_reg_loss = 0
        with torch.no_grad():
            for data in val_iterator:
                if isinstance(data, list):
                    x = data[0]
                else:
                    x = data

                x = x.to(device)

                val_loss, val_score_loss, val_reg_loss = compute_loss_variance_reduction(phi, psi, x, config.std, train=False, 
                                                                       use_cv=config.use_cv, use_reg=config.use_reg, reg_factor=config.reg_factor, device=device)

                total_val_loss += val_loss.item()
                total_val_score_loss += val_score_loss.item()
                if val_reg_loss is not None:
                    total_val_reg_loss += val_reg_loss.item()
                val_iterator.set_postfix(val_loss=val_loss.item())

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_score_loss = total_val_score_loss / len(val_loader)
        avg_val_reg_loss = total_val_reg_loss / len(val_loader) if total_val_reg_loss > 0 else 0
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Loss/Score Validation", avg_val_score_loss, epoch)
        writer.add_scalar("Loss/Regularization Validation", avg_val_reg_loss, epoch)

        print(f"Epoch {epoch+1}/{config.epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        if (epoch + 1) % config.eval_log_frequency == 0:
            check_manifold_properties(phi, psi, writer, epoch, device, val_loader)
        
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
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    main(args.config)
