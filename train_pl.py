import pytorch_lightning as pl
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from src.diffeomorphisms import get_diffeomorphism
from src.diffeomorphisms.utils import get_principal_components
from src.strongly_convex.learnable_psi import LearnablePsi
from src.training.train_utils import EMA, load_config, set_visible_gpus, set_seed
from src.training.optim_utils import LightningWarmUpCosineAnnealingScheduler
from src.training.loss import get_loss_function
from src.data import get_dataset
from src.training.callbacks import check_manifold_properties
from torch.optim.lr_scheduler import CosineAnnealingLR

# Set which GPUs are visible
set_visible_gpus('2')

class DiffModel(pl.LightningModule):
    def __init__(self, config):
        super(DiffModel, self).__init__()
        self.config = config

        # Set random seeds
        set_seed(config.seed)

        # Initialize the models
        U, mean = None, None
        if config.get('premultiplication_by_U', False):
            train_loader = self.train_dataloader()
            U, mean = get_principal_components(train_loader)

        self.phi = get_diffeomorphism(config, U=U, mean=mean)
        self.psi = LearnablePsi(config.d)

        self.ema_phi = EMA(model=self.phi, decay=0.999)
        self.ema_psi = EMA(model=self.psi, decay=0.999)

        # Loss function
        self.loss_fn = get_loss_function(config)

        # Logging the number of parameters
        self.phi_total_params, self.phi_trainable_params = self.count_parameters(self.phi)
        self.psi_total_params, self.psi_trainable_params = self.count_parameters(self.psi)
        print(f"Model Phi - Total Parameters: {self.phi_total_params}, Trainable Parameters: {self.phi_trainable_params}")
        print(f"Model Psi - Total Parameters: {self.psi_total_params}, Trainable Parameters: {self.psi_trainable_params}")

    def on_fit_start(self):
        # Move model and EMA shadow weights to the correct device
        self.phi.to(self.device)
        self.psi.to(self.device)

        # Move EMA shadow weights to the correct device
        for name in self.ema_phi.shadow:
            self.ema_phi.shadow[name] = self.ema_phi.shadow[name].to(self.device)
        for name in self.ema_psi.shadow:
            self.ema_psi.shadow[name] = self.ema_psi.shadow[name].to(self.device)

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, list) else batch
        loss, density_learning_loss, reg_loss = self.loss_fn(self.phi, self.psi, x, train=True, device=self.device)
        
        # Log losses
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_density_learning_loss", density_learning_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if reg_loss is not None:
            self.log("train_reg_loss", reg_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Log the learning rate
        optimizer = self.optimizers()  # Retrieve the optimizer
        lr = optimizer.param_groups[0]['lr']  # Get the learning rate of the first param group
        self.log("learning_rate", lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, list) else batch
        loss, density_learning_loss, reg_loss = self.loss_fn(self.phi, self.psi, x, train=False, device=self.device)

        # Log losses
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_density_learning_loss", density_learning_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if reg_loss is not None:
            self.log("val_reg_loss", reg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self):
        # Setup the optimizer
        optimizer_type = getattr(self.config, 'optimizer', 'Adam')
        learning_rate = self.config.get('learning_rate', 2e-4)

        if optimizer_type == 'AdamW':
            optimizer = optim.AdamW(
                list(self.phi.parameters()) + list(self.psi.parameters()),
                lr=learning_rate,
                betas=(self.config.get('beta1', 0.9), self.config.get('beta2', 0.99)),
                eps=self.config.get('eps', 1e-8),
                weight_decay=self.config.get('weight_decay', 0.01)
            )
        elif optimizer_type == 'Adam':
            optimizer = optim.Adam(
                list(self.phi.parameters()) + list(self.psi.parameters()),
                lr=learning_rate,
                betas=(self.config.get('beta1', 0.9), self.config.get('beta2', 0.99)),
                eps=self.config.get('eps', 1e-8),
                weight_decay=self.config.get('weight_decay', 1e-5)
            )
        elif optimizer_type == 'RMSprop':
            optimizer = optim.RMSprop(
                list(self.phi.parameters()) + list(self.psi.parameters()),
                lr=learning_rate,
                alpha=self.config.get('alpha', 0.99),
                eps=self.config.get('eps', 1e-8),
                weight_decay=self.config.get('weight_decay', 1e-5)
            )
        elif optimizer_type == 'SGD':
            optimizer = optim.SGD(
                list(self.phi.parameters()) + list(self.psi.parameters()),
                lr=learning_rate,
                momentum=self.config.get('momentum', 0.9),
                weight_decay=self.config.get('weight_decay', 1e-5)
            )
        else:
            raise ValueError(f"Optimizer {optimizer_type} is not supported.")

        # Setup the cosine annealing scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.epochs * len(self.train_dataloader()),  # Total number of steps (T_max)
            eta_min=self.config.get('eta_min', 0)  # Minimum learning rate at the end of scheduling
        )

        # Configure the scheduler to be called at every epoch
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',  # Adjust the learning rate every step
            'frequency': 1,      # Call the scheduler every step
        }

        return [optimizer], [scheduler_config]


    def train_dataloader(self):
        dataset_class = get_dataset(self.config.dataset_class)
        train_dataset = dataset_class(self.config, split='train')
        return DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

    def val_dataloader(self):
        dataset_class = get_dataset(self.config.dataset_class)
        val_dataset = dataset_class(self.config, split='val')
        return DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

    def on_validation_epoch_start(self):
        # Apply EMA weights before starting validation
        self.ema_phi.apply_shadow()
        self.ema_psi.apply_shadow()

    def on_validation_epoch_end(self):
        # Restore original weights after validation
        self.ema_phi.restore()
        self.ema_psi.restore()

    def optimizer_step(self, *args, **kwargs):
        # Perform the optimizer step
        super().optimizer_step(*args, **kwargs)

        # Update EMA after the optimizer step
        self.ema_phi.update()
        self.ema_psi.update()

    def on_save_checkpoint(self, checkpoint):
        # Manually save the EMA shadow weights
        checkpoint['ema_phi_shadow'] = self.ema_phi.shadow
        checkpoint['ema_psi_shadow'] = self.ema_psi.shadow

    def on_load_checkpoint(self, checkpoint):
        # Manually load the EMA shadow weights
        self.ema_phi.shadow = checkpoint['ema_phi_shadow']
        self.ema_psi.shadow = checkpoint['ema_psi_shadow']


    @staticmethod
    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params


class ManifoldPropertiesCallback(pl.Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.eval_log_frequency = config.eval_log_frequency

    def on_validation_epoch_end(self, trainer, pl_module):
        # Execute the manifold check only at the specified frequency
        if (trainer.current_epoch + 1) % self.eval_log_frequency == 0:
            check_manifold_properties(
                self.config.dataset,
                pl_module.phi,
                pl_module.psi,
                trainer.logger.experiment,  # This is equivalent to the TensorBoard writer
                trainer.current_epoch,
                pl_module.device,
                pl_module.val_dataloader(),
                self.config.d
            )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training with Config File")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)

    # Handle logging directories
    tensorboard_dir = os.path.join(config.base_log_dir, config.experiment, 'training_logs')
    checkpoint_dir = os.path.join(config.base_log_dir, config.experiment, 'checkpoints')

    # Set up the logger
    logger = TensorBoardLogger(save_dir=tensorboard_dir, name="")

    # Initialize the model
    model = DiffModel(config)

    # Initialize the custom callback
    manifold_properties_callback = ManifoldPropertiesCallback(config)

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,  # Save only the best 3 models based on validation loss
        monitor="val_loss",
        mode="min",
        save_weights_only=False,
        every_n_epochs=1,  # Save a checkpoint every epoch
        save_last=True  # Also save the last checkpoint
    )

    # Trainer setup with the callback and logger
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=config.epochs,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        callbacks=[manifold_properties_callback, checkpoint_callback],  # Add the custom callbacks here
        gradient_clip_val=1.0,
        num_sanity_val_steps=2,
        logger=logger  # Use the TensorBoard logger
    )

    if config.checkpoint:
        trainer.fit(model, ckpt_path=config.checkpoint)
    else:
        trainer.fit(model)
