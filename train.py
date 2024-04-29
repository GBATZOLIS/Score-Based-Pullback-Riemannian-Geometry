import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
from src.diffeomorphisms.EuclideanDiffeomorphism import EuclideanDiffeomorphism
from src.strongly_convex.learnable_psi import LearnablePsi
import os
from src.training.train_utils import EMA, load_model, save_model, get_log_density_fn, get_score_fn, WarmUpScheduler

def main(args):
    writer = SummaryWriter(log_dir=args.tensorboard_dir)

    # Initialize the models
    phi = EuclideanDiffeomorphism(args)
    psi = LearnablePsi(args.d)

    # Initialize EMA handlers
    ema_phi = EMA(model=phi, decay=0.999)
    ema_psi = EMA(model=psi, decay=0.999)

    # DataLoader for synthetic training and validation data
    dataset = TensorDataset(torch.randn(1000, args.d))
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = TensorDataset(torch.randn(200, args.d))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Optimizer and scheduler
    optimizer = optim.Adam(list(phi.parameters()) + list(psi.parameters()), lr=0)
    scheduler = WarmUpScheduler(optimizer, args.learning_rate, warmup_steps=1000)

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

    # Training and Validation loop
    for epoch in range(args.epochs):
        phi.train()
        psi.train()
        total_loss = 0
        for data in train_loader:
            x = data[0].to(device)
            std = args.std * torch.ones_like(x)
            optimizer.zero_grad()
            score_fn = get_score_fn(phi, psi)
            z = torch.randn_like(x)
            x_pert = x + std * z
            score = score_fn(x_pert)
            grad_log_pert_kernel = -1 * z / std
            target = grad_log_pert_kernel
            loss = torch.mean((score - target)**2)
            loss.backward()

            # Update EMA and apply clipping
            ema_phi.update()
            ema_psi.update()
            torch.nn.utils.clip_grad_norm_(list(phi.parameters()) + list(psi.parameters()), max_norm=1)
            optimizer.step()
            scheduler.step()

            # Logging learning rate every 10 steps
            if step % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("Learning Rate", current_lr, step)

            step += 1
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)

        # Validation
        ema_phi.apply_shadow()
        ema_psi.apply_shadow()

        phi.eval()
        psi.eval()
        with torch.no_grad():
            total_val_loss = 0
            for data in val_loader:
                x = data[0].to(device)
                std = args.std * torch.ones_like(x)
                score_fn = get_score_fn(phi, psi)
                z = torch.randn_like(x)
                x_pert = x + std * z
                score = score_fn(x_pert)
                grad_log_pert_kernel = -1 * z / std
                target = grad_log_pert_kernel
                val_loss = torch.mean((score - target)**2)
                total_val_loss += val_loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            writer.add_scalar("Loss/Validation", avg_val_loss, epoch)

        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        ema_phi.restore()
        ema_psi.restore()

        #save the model weights (both running weights and EMA weights)
        if (epoch + 1) % args.checkpoint_frequency == 0:
            save_model(phi, ema_phi, epoch, loss, "Phi", args.checkpoint_dir)
            save_model(psi, ema_psi, epoch, loss, "Psi", args.checkpoint_dir)


    writer.close()
    print("Training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training with Checkpointing for Separate Models")

    # Model settings
    parser.add_argument("--base_transform_type", type=str, default="rq", choices=["rq", "affine"], help="Type of base transform: 'rq' for Rational Quadratic, 'affine' for Affine.")
    parser.add_argument("--hidden_features", type=int, default=64, help="Number of hidden features in the model.")
    parser.add_argument("--num_transform_blocks", type=int, default=2, help="Number of blocks in each transform layer.")
    parser.add_argument("--num_flow_steps", type=int, default=2, help="Number of flow steps in the diffeomorphism.")
    parser.add_argument("--use_batch_norm", type=int, default=0, choices=[0, 1], help="Whether to use batch normalization.")
    parser.add_argument("--dropout_probability", type=float, default=0.0, help="Dropout probability in the network.")

    # Training settings
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints.")
    parser.add_argument("--checkpoint_frequency", type=int, default=1, help="Frequency of saving the model per number of epochs.")
    parser.add_argument("--std", type=float, default=1e-3, help="Perturbation std for denoising score matching")
    
    # Data handling
    parser.add_argument("--d", type=int, default=2, help="Dimensionality of the input space.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")

    # Device settings
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for computation (e.g., 'cpu', 'cuda').")

    # Optimization settings
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for optimizer.")

    # Optional loading of model checkpoints for resuming
    parser.add_argument("--load_phi_checkpoint", type=str, default=None, help="Path to load phi model checkpoint if any.")
    parser.add_argument("--load_psi_checkpoint", type=str, default=None, help="Path to load psi model checkpoint if any.")

    # TensorBoard logging directory
    parser.add_argument("--tensorboard_dir", type=str, default="./logs", help="Directory to save TensorBoard logs.")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=1638128, help="Random seed for PyTorch and NumPy.")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)

