import torch
import numpy as np
import os
import sys
import importlib.util
from collections import defaultdict


class EMA:
    """
    Exponential Moving Average (EMA) for maintaining a moving average of model weights.

    Args:
        model (torch.nn.Module): The model for which EMA will be applied.
        decay (float): Decay rate for updating the EMA weights. Typically a value close to 1.

    Attributes:
        model (torch.nn.Module): The model to which EMA is applied.
        decay (float): Decay rate for updating the EMA weights.
        shadow (dict): Dictionary containing the shadow variables for EMA.
        backup (dict): Dictionary containing the backup of model parameters.

    Methods:
        __init__(model, decay):
            Initializes the EMA with the provided model and decay rate.
            Creates shadow variables for each trainable parameter in the model.

        update():
            Updates the shadow variables with the current model parameters.
            This is typically called after each optimization step.

        apply_shadow():
            Applies the shadow variables to the model parameters.
            This is typically called before evaluation or saving checkpoints.

        restore():
            Restores the original model parameters from the backup.
            This is typically called after evaluation or saving checkpoints.
    """
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize the shadow variables with model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(param.data.device)

    def update(self):
        """
        Updates the shadow variables with the current model parameters.
        This is typically called after each optimization step.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.shadow[name].data = (1.0 - self.decay) * param.data + self.decay * self.shadow[name].data

    def apply_shadow(self):
        """
        Applies the shadow variables to the model parameters.
        This is typically called before evaluation or saving checkpoints.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].data.clone()

    def restore(self):
        """
        Restores the original model parameters from the backup.
        This is typically called after evaluation or saving checkpoints.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name].clone()
        self.backup = {}

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
    
class WarmUpScheduler:
    def __init__(self, optimizer, target_lr, warmup_steps):
        self.optimizer = optimizer
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.target_lr * min(1.0, self.step_num / self.warmup_steps)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


    
def save_model(phi, ema_phi, psi, ema_psi, epoch, loss, checkpoint_dir, best_checkpoints, global_step, best_val_loss, epochs_no_improve, optimizer, scheduler):
    def write_model(state_dict, path, epoch, loss, global_step, best_checkpoints, best_val_loss, epochs_no_improve):
        scheduler.step_num = global_step
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': state_dict,
            'loss': loss,
            'global_step': global_step,
            'best_checkpoints': best_checkpoints,
            'best_val_loss': best_val_loss,
            'epochs_no_improve': epochs_no_improve,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
        }
        torch.save(checkpoint, path)
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Save the latest model checkpoint
    last_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_last.pth")
    state_dict = {'phi': phi.state_dict(), 'psi': psi.state_dict()}
    write_model(state_dict, last_checkpoint_path, epoch, loss, global_step, best_checkpoints, best_val_loss, epochs_no_improve)

    # Save the latest EMA model checkpoint
    last_ema_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_last_EMA.pth")
    ema_state_dict = {'phi': ema_phi.shadow, 'psi': ema_psi.shadow}
    write_model(ema_state_dict, last_ema_checkpoint_path, epoch, loss, global_step, best_checkpoints, best_val_loss, epochs_no_improve)
    
    # Manage the best checkpoints
    if len(best_checkpoints) < 3:
        new_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_loss_{loss:.3f}.pth")
        new_ema_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_loss_{loss:.3f}_EMA.pth")
        best_checkpoints.append((new_checkpoint_path, new_ema_checkpoint_path, loss))
        write_model(state_dict, new_checkpoint_path, epoch, loss, global_step, best_checkpoints, best_val_loss, epochs_no_improve)
        write_model(ema_state_dict, new_ema_checkpoint_path, epoch, loss, global_step, best_checkpoints, best_val_loss, epochs_no_improve)
    else:
        worst_checkpoint = max(best_checkpoints, key=lambda x: x[2])
        if loss < worst_checkpoint[2]:
            best_checkpoints.remove(worst_checkpoint)
            if os.path.exists(worst_checkpoint[0]):
                os.remove(worst_checkpoint[0])
            if os.path.exists(worst_checkpoint[1]):
                os.remove(worst_checkpoint[1])

            new_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_loss_{loss:.3f}.pth")
            new_ema_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_loss_{loss:.3f}_EMA.pth")
            best_checkpoints.append((new_checkpoint_path, new_ema_checkpoint_path, loss))
            write_model(state_dict, new_checkpoint_path, epoch, loss, global_step, best_checkpoints, best_val_loss, epochs_no_improve)
            write_model(ema_state_dict, new_ema_checkpoint_path, epoch, loss, global_step, best_checkpoints, best_val_loss, epochs_no_improve)
    
    print(f"Model saved at '{last_checkpoint_path}'")
    print(f"EMA model saved at '{last_ema_checkpoint_path}'")

def load_model(checkpoint_path, phi, ema_phi, psi, ema_psi, optimizer=None, scheduler=None, is_ema=False):
    checkpoint = torch.load(checkpoint_path)
    if is_ema:
        ema_phi.shadow = checkpoint['model_state_dict']['phi']
        ema_psi.shadow = checkpoint['model_state_dict']['psi']
    else:
        phi.load_state_dict(checkpoint['model_state_dict']['phi'])
        psi.load_state_dict(checkpoint['model_state_dict']['psi'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    global_step = checkpoint.get('global_step', 0)
    best_checkpoints = checkpoint.get('best_checkpoints', [])
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Model {'EMA ' if is_ema else ''}loaded from '{checkpoint_path}', Epoch: {epoch}, Loss: {loss}")
    
    return epoch, loss, global_step, best_checkpoints, best_val_loss, epochs_no_improve

def save_model_weights_for_comparison(model):
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.data.clone()
    return weights

def compute_l2_distance(weights1, weights2):
    """ Compute the L2 distance between two sets of weights. """
    vector1 = torch.cat([w.flatten() for w in weights1.values()])
    vector2 = torch.cat([w.flatten() for w in weights2.values()])
    distance = torch.norm(vector1 - vector2).item()
    return distance

def resume_training(config, phi, ema_phi, psi, ema_psi, load_model_func, get_optimizer_and_scheduler_func, total_steps, train_loader):
    checkpoint_path = config.checkpoint
    if checkpoint_path:
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(config.base_log_dir, config.experiment, 'checkpoints', checkpoint_path)
        if not checkpoint_path.endswith('.pth'):
            checkpoint_path += '.pth'
        
        optimizer, scheduler = get_optimizer_and_scheduler_func(list(phi.parameters()) + list(psi.parameters()), config, total_steps)

        # Save original weights for comparison
        original_phi_weights = save_model_weights_for_comparison(phi)
        original_psi_weights = save_model_weights_for_comparison(psi)

        # Load models
        epoch, loss, global_step, best_checkpoints, best_val_loss, epochs_no_improve = load_model_func(
            checkpoint_path, phi, ema_phi, psi, ema_psi, optimizer, scheduler, is_ema=False
        )
        ema_checkpoint_path = checkpoint_path.replace('.pth', '_EMA.pth')
        load_model_func(ema_checkpoint_path, phi, ema_phi, psi, ema_psi, is_ema=True)

        # Compare weights after loading
        phi_l2_distance = compute_l2_distance(original_phi_weights, save_model_weights_for_comparison(phi))
        psi_l2_distance = compute_l2_distance(original_psi_weights, save_model_weights_for_comparison(psi))
        print(f"L2 distance for Phi model: {phi_l2_distance}")
        print(f"L2 distance for Psi model: {psi_l2_distance}")
        print(f"Resuming training from epoch {epoch + 1}")        
        return epoch + 1, global_step, best_checkpoints, best_val_loss, epochs_no_improve, optimizer, scheduler
    else:
        optimizer, scheduler = get_optimizer_and_scheduler_func(list(phi.parameters()) + list(psi.parameters()), config, total_steps)
        return 0, 0, [], float('inf'), 0, optimizer, scheduler


def get_log_density_fn(phi, psi):
    def log_density_fn(x):
        phi_x = phi.forward(x)
        return -psi.forward(phi_x)
    return log_density_fn

def get_score_fn(phi, psi, train=True):
    log_density_fn = get_log_density_fn(phi, psi)
    
    def score_fn(x):
        if not train:
            torch.set_grad_enabled(True)
        x.requires_grad_(True)  # Ensure x requires gradient
        log_density = log_density_fn(x)
        grad_log_density = torch.autograd.grad(log_density.sum(), x, create_graph=True)[0]
        if not train:
            torch.set_grad_enabled(False)
        return grad_log_density
    return score_fn

def set_visible_gpus(gpus):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

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

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config_module"] = config_module
    spec.loader.exec_module(config_module)
    return config_module.get_config()

def get_full_checkpoint_path(checkpoint_dir, filename):
    return os.path.join(checkpoint_dir, filename) if not os.path.isabs(filename) else filename