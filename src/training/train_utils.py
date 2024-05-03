import torch
import os 


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
                self.shadow[name] = param.data.clone()

    def update(self):
        """
        Updates the shadow variables with the current model parameters.
        This is typically called after each optimization step.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """
        Applies the shadow variables to the model parameters.
        This is typically called before evaluation or saving checkpoints.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """
        Restores the original model parameters from the backup.
        This is typically called after evaluation or saving checkpoints.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


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


    
def save_model(model, ema_model, epoch, loss, model_name, checkpoint_dir, best_checkpoints):
    def write_model(model, path, epoch, loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': loss,
        }, path)
    
        
    """
    Saves the model checkpoints including the last and the top three based on the loss.
    
    Args:
        model (torch.nn.Module): The actual model to save.
        ema_model (EMA): The EMA handler containing the shadow weights.
        epoch (int): Current epoch number.
        loss (float): Loss at the current epoch.
        model_name (str): Name of the model.
        checkpoint_dir (str): Directory path to save the checkpoint.
        best_checkpoints (list): List storing the top 3 checkpoints based on loss.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Save the last model state
    last_checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_last.pth")
    write_model(model, last_checkpoint_path, epoch, loss)

    last_ema_checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_last_EMA.pth")
    write_model(model, last_ema_checkpoint_path, epoch, loss)
    
    # Update the list of best checkpoints
    if len(best_checkpoints) < 3:
        new_checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch}_loss_{loss:.3f}.pth")
        new_ema_checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch}_loss_{loss:.3f}_EMA.pth")
        best_checkpoints.append((new_checkpoint_path, new_ema_checkpoint_path, loss))
        write_model(model, new_checkpoint_path, epoch, loss)
        write_model(model, new_ema_checkpoint_path, epoch, loss)
    else:
        # Find the worst checkpoint (highest loss)
        worst_checkpoint = max(best_checkpoints, key=lambda x: x[2])
        if loss < worst_checkpoint[2]:
            # Replace the worst checkpoint
            best_checkpoints.remove(worst_checkpoint)
            os.remove(worst_checkpoint[0])
            os.remove(worst_checkpoint[1])

            new_checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch}_loss_{loss:.3f}.pth")
            new_ema_checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch}_loss_{loss:.3f}_EMA.pth")
            best_checkpoints.append((new_checkpoint_path, new_ema_checkpoint_path, loss))
            
            write_model(model, new_checkpoint_path, epoch, loss)
            write_model(model, new_ema_checkpoint_path, epoch, loss)
    
            print(f"{model_name} model saved at '{new_checkpoint_path}'")
            print(f"{model_name} EMA model saved at '{new_ema_checkpoint_path}'")



def load_model(model, ema_model, checkpoint_path, model_name, is_ema=False):
    """
    Loads a single model and optionally its EMA from a checkpoint file.

    Args:
        model (torch.nn.Module): The model to load state into.
        ema_model (EMA): The EMA handler for the model.
        checkpoint_path (str): Path to the checkpoint file.
        model_name (str): Name of the model (used for logging).
        is_ema (bool): Flag to indicate if the checkpoint contains EMA weights.

    Returns:
        int: The epoch number of the checkpoint.
        float: The loss at the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path)
    if is_ema:
        ema_model.shadow = {name: torch.tensor(data) for name, data in checkpoint['model_state_dict'].items()}
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f"{model_name} {'EMA' if is_ema else ''} model loaded from '{checkpoint_path}', Epoch: {epoch}, Loss: {loss}")
    return epoch, loss


def get_log_density_fn(phi, psi):
    def log_density_fn(x):
        return -psi.forward(phi.forward(x))
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

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

