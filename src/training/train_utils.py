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

def save_model(model, ema_model, epoch, loss, model_name, checkpoint_dir):
    """
    Saves a single model and its EMA counterpart to a checkpoint file.

    Args:
        model (torch.nn.Module): The actual model to save.
        ema_model (EMA): The EMA handler containing the shadow weights.
        epoch (int): Current epoch number.
        loss (float): Loss at the current epoch.
        model_name (str): Name of the model (used in filename).
        checkpoint_dir (str): Directory path to save the checkpoint.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch}.pth")
    ema_checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_EMA_epoch_{epoch}.pth")

    # Save actual model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }, checkpoint_path)

    # Save EMA model
    torch.save({
        'epoch': epoch,
        'model_state_dict': ema_model.shadow,
        'loss': loss,
    }, ema_checkpoint_path)

    print(f"{model_name} model saved at '{checkpoint_path}'")
    print(f"{model_name} EMA model saved at '{ema_checkpoint_path}'")


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

def get_score_fn(phi, psi):
    log_density_fn = get_log_density_fn(phi, psi)
    
    def score_fn(x):
        x.requires_grad_(True)  # Ensure x requires gradient
        log_density = log_density_fn(x)
        grad_log_density = torch.autograd.grad(log_density.sum(), x, create_graph=True)[0]
        return grad_log_density
    return score_fn
