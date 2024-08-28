import torch.optim as optim
import math
import torch.optim.lr_scheduler as lr_scheduler

class LightningWarmUpCosineAnnealingScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, target_lr, warmup_steps, total_steps, current_step=-1):
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch=current_step)

    def get_lr(self):
        current_step = self.last_epoch + 1  # Increment step count as get_lr is called
        if current_step < self.warmup_steps:
            # Linearly increase the learning rate during the warmup period
            lr = self.target_lr * (current_step / self.warmup_steps)
        else:
            # After the warmup period, apply cosine annealing
            progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = 0.5 * self.target_lr * (1 + math.cos(math.pi * progress))

        # Return the learning rate for each parameter group
        return [lr for _ in self.base_lrs]

    def state_dict(self):
        # Save the current step (last_epoch) along with custom state
        return {
            'target_lr': self.target_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'current_step': self.last_epoch,
            **super().state_dict()
        }

    def load_state_dict(self, state_dict):
        # Load the custom state, including the current step (last_epoch)
        self.target_lr = state_dict.pop('target_lr')
        self.warmup_steps = state_dict.pop('warmup_steps')
        self.total_steps = state_dict.pop('total_steps')
        self.last_epoch = state_dict.pop('current_step')
        super().load_state_dict(state_dict)

class WarmUpCosineAnnealingScheduler:
    def __init__(self, optimizer, target_lr, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.target_lr * min(1.0, self.step_num / self.warmup_steps)
        else:
            progress = (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = 0.5 * self.target_lr * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        return {
            'step_num': self.step_num,
            'optimizer_state': self.optimizer.state_dict()
        }

    def load_state_dict(self, state):
        self.step_num = state['step_num']
        self.optimizer.load_state_dict(state['optimizer_state'])


def get_optimizer_and_scheduler(model_params, config, total_steps):
    """
    Sets up the optimizer and scheduler based on the configuration provided.
    
    Args:
        model_params: The parameters of the model(s) to be optimized.
        config: Configuration dictionary containing optimization settings.
        total_steps: Total number of training steps.
    
    Returns:
        optimizer: The initialized optimizer.
        scheduler: The initialized scheduler.
    """
    optimizer_type = getattr(config, 'optimizer', 'Adam')

    if optimizer_type == 'AdamW':
        optimizer = optim.AdamW(
            model_params,
            lr=config.get('learning_rate', 2e-4),
            betas=(config.get('beta1', 0.9), config.get('beta2', 0.99)),
            eps=config.get('eps', 1e-8),
            weight_decay=config.get('weight_decay', 0.01)
        )
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(
            model_params,
            lr=config.get('learning_rate', 2e-4),
            betas=(config.get('beta1', 0.9), config.get('beta2', 0.99)),
            eps=config.get('eps', 1e-8),
            weight_decay=config.get('weight_decay', 1e-5)
        )
    elif optimizer_type == 'RMSprop':
        optimizer = optim.RMSprop(
            model_params,
            lr=config.get('learning_rate', 1e-4),
            alpha=config.get('alpha', 0.99),
            eps=config.get('eps', 1e-8),
            weight_decay=config.get('weight_decay', 1e-5)
        )
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(
            model_params,
            lr=config.get('learning_rate', 1e-2),
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 1e-5)
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} is not supported.")
    
    # Setup the scheduler
    scheduler = WarmUpCosineAnnealingScheduler(
        optimizer, 
        config.get('learning_rate', 2e-4), 
        warmup_steps=config.get('warmup_steps', 1000),
        total_steps=total_steps
    )
    
    return optimizer, scheduler
