import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
from src.diffeomorphisms.euclidean_diffeomorphism import EuclideanDiffeomorphism
from src.strongly_convex.learnable_psi import LearnablePsi
from src.data.numpy_dataset import NumpyDataset
import os
from src.training.train_utils import EMA, load_model, save_model, get_log_density_fn, get_score_fn, WarmUpScheduler, count_parameters
from src.training.callbacks import check_manifold_properties
from tqdm import tqdm 

from torch.autograd.functional import jvp

def compute_loss(phi, psi, x, args_std, train=True, use_reg=False, reg_factor=1, device='cuda:0'):
    def score_matching_loss():
        std = args_std * torch.ones_like(x)
        score_fn = get_score_fn(phi, psi, train=train)
        z = torch.randn_like(x)
        x_pert = x + std * z
        score = score_fn(x_pert)
        #grad_log_pert_kernel = -1 * z / std
        #target = grad_log_pert_kernel
        target = -1 * z
        loss = torch.mean((std * score - target)**2)
        return loss

    def regularization_term():
        batch_size, dim = x.shape
        # Generate random unit norm vectors for each batch element
        v = torch.randn(batch_size, dim, device=device)
        v /= v.norm(dim=1, keepdim=True)

        if not train:
            torch.set_grad_enabled(True)

        # Compute the Jacobian-vector product using PyTorch's jvp function
        _, Jv = jvp(phi, (x,), (v,))

        if not train:
            torch.set_grad_enabled(False)
        
        # Calculate the norm of each vector in the batch and their deviation from 1
        norms = Jv.norm(dim=1)
        return torch.mean((norms - 1) ** 2)
    
    if use_reg:
        loss = score_matching_loss() + reg_factor*regularization_term()
    else:
        loss = score_matching_loss()
    
    return loss

def main(args):
    #handle logging directories
    tensorboard_dir = os.path.join(args.base_log_dir, args.experiment, 'training_logs')
    checkpoint_dir = os.path.join(args.base_log_dir, args.experiment, 'checkpoints')

    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Initialize the models
    phi = EuclideanDiffeomorphism(args)
    psi = LearnablePsi(args.d)

    # Print model summaries
    phi_total_params, phi_trainable_params = count_parameters(phi)
    psi_total_params, psi_trainable_params = count_parameters(psi)
    print(f"Model Phi - Total Parameters: {phi_total_params}, Trainable Parameters: {phi_trainable_params}")
    print(f"Model Psi - Total Parameters: {psi_total_params}, Trainable Parameters: {psi_trainable_params}")

    # Initialize EMA handlers
    ema_phi = EMA(model=phi, decay=0.999)
    ema_psi = EMA(model=psi, decay=0.999)

    # DataLoader for synthetic training and validation data
    dataset = NumpyDataset(args, split='train')
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = NumpyDataset(args, split='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Optimizer and scheduler
    optimizer = optim.Adam(list(phi.parameters()) + list(psi.parameters()), lr=0)
    scheduler = WarmUpScheduler(optimizer, args.learning_rate, warmup_steps=2000)

    # Device configuration
    device = torch.device(args.device)
    phi.to(device)
    psi.to(device)

    # Load checkpoints if specified
    if args.load_phi_checkpoint:
        epoch_phi, loss_phi = load_model(phi, None, args.load_phi_checkpoint, "Phi")
        ema_path_phi = args.load_phi_checkpoint.replace('.pth', '_EMA.pth')
        load_model(phi, ema_phi, ema_path_phi, "Phi EMA", is_ema=True)

    if args.load_psi_checkpoint:
        epoch_psi, loss_psi = load_model(psi, None, args.load_psi_checkpoint, "Psi")
        ema_path_psi = args.load_psi_checkpoint.replace('.pth', '_EMA.pth')
        load_model(psi, ema_psi, ema_path_psi, "Psi EMA", is_ema=True)

    step = 0
    best_checkpoints = []  # To track the top K checkpoints based on loss

    # Training and Validation loop
    for epoch in range(args.epochs):
        phi.train()
        psi.train()

        train_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{args.epochs}", leave=False)
        total_loss = 0
        for data in train_iterator:
            x = data.to(device)
            loss = compute_loss(phi, psi, x, args.std, train=True, use_reg=args.use_reg, reg_factor=args.reg_factor, device=device)

            '''
            std = args.std * torch.ones_like(x)
            optimizer.zero_grad()
            score_fn = get_score_fn(phi, psi, train=True)
            z = torch.randn_like(x)
            x_pert = x + std * z
            score = score_fn(x_pert)
            #grad_log_pert_kernel = -1 * z / std
            #target = grad_log_pert_kernel
            target = -1 * z
            loss = torch.mean((std * score - target)**2)
            '''
            loss.backward()

            # Apply clipping before optimizer step
            torch.nn.utils.clip_grad_norm_(list(phi.parameters()) + list(psi.parameters()), max_norm=1)
            optimizer.step()

            # Update EMA after the optimizer step
            ema_phi.update()
            ema_psi.update()

            scheduler.step()

            # Logging learning rate every 10 steps
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
        val_iterator = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{args.epochs}", leave=False)
        total_val_loss = 0
        with torch.no_grad():
            for data in val_iterator:
                x = data.to(device)
                val_loss = compute_loss(phi, psi, x, args.std, train=False, use_reg=args.use_reg, reg_factor=args.reg_factor, device=device)

                '''
                std = args.std * torch.ones_like(x)
                score_fn = get_score_fn(phi, psi, train=False)
                z = torch.randn_like(x)
                x_pert = x + std * z
                score = score_fn(x_pert)
                #grad_log_pert_kernel = -1 * z / std
                #target = grad_log_pert_kernel
                target = -1 * z
                val_loss = torch.mean((std * score - target)**2)
                '''

                total_val_loss += val_loss.item()
                val_iterator.set_postfix(val_loss=val_loss.item())

        avg_val_loss = total_val_loss / len(val_loader)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)

        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # Inside your main training loop:
        if (epoch + 1) % args.eval_log_frequency == 0:  # Adjust the interval as needed
            check_manifold_properties(phi, psi, writer, epoch)
        
        ema_phi.restore()
        ema_psi.restore()
        
        #save the model weights (both running weights and EMA weights)
        if (epoch + 1) % args.checkpoint_frequency == 0:
            save_model(phi, ema_phi, epoch, avg_train_loss, "Phi", checkpoint_dir, best_checkpoints)
            save_model(psi, ema_psi, epoch, avg_train_loss, "Psi", checkpoint_dir, best_checkpoints)


    writer.close()
    print("Training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training with Checkpointing for Separate Models")

    #logging settings
    parser.add_argument("--base_log_dir", type=str, default="./results", help="Directory to save TensorBoard logs.")
    parser.add_argument("--experiment", type=str, default="single_banana", help="Directory to save logs.")
    parser.add_argument("--eval_log_frequency", type=int, default=200, help="number of epochs between successive evaluations of the method")

    # Model settings
    parser.add_argument('--base_transform_type', type=str, default='affine', help='Which base transform to use.')
    parser.add_argument('--hidden_features', type=int, default=64, help='Number of hidden features in coupling layers.')
    parser.add_argument('--num_transform_blocks', type=int, default=2, help='Number of blocks in coupling layer transform.')
    parser.add_argument('--use_batch_norm', type=int, default=0, choices=[0, 1], help='Whether to use batch norm in coupling layer transform.')
    parser.add_argument('--dropout_probability', type=float, default=0.0, help='Dropout probability for coupling transform.')
    parser.add_argument('--num_bins', type=int, default=128, help='Number of bins in piecewise cubic coupling transform.')
    parser.add_argument('--apply_unconditional_transform', type=int, default=0, choices=[0, 1], help='Whether to apply unconditional transform in coupling layer.')
    parser.add_argument('--min_bin_width', type=float, default=1e-3, help='Minimum bin width for piecewise transforms.')
    parser.add_argument('--num_flow_steps', type=int, default=2, help='Number of steps of flow.')

    # Training settings
    parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs to train.")
    parser.add_argument("--checkpoint_frequency", type=int, default=1, help="Frequency of saving the model per number of epochs.")
    parser.add_argument("--std", type=float, default=1e-1, help="Perturbation std for denoising score matching")
    parser.add_argument("--use_reg", action='store_true', help="Whether to use regularization for enforcing local isometry.")
    parser.add_argument("--reg_factor", type=float, default=1, help="Factor to scale the regularization term.")

    # Data handling
    parser.add_argument("--data_path", type=str, default="./data", help="Data directory.")
    parser.add_argument("--d", type=int, default=2, help="Dimensionality of the input space.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")

    # Device settings
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for computation (e.g., 'cpu', 'cuda').")

    # Optimization settings
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate for optimizer.")

    # Optional loading of model checkpoints for resuming
    parser.add_argument("--load_phi_checkpoint", type=str, default=None, help="Path to load phi model checkpoint if any.")
    parser.add_argument("--load_psi_checkpoint", type=str, default=None, help="Path to load psi model checkpoint if any.")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=1638128, help="Random seed for PyTorch and NumPy.")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)

