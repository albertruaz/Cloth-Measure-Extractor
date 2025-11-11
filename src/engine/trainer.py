"""Trainer for garment landmark regression."""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import logging
from tqdm import tqdm
import numpy as np

from ..utils.metrics import compute_pck, compute_mse
from ..utils.heatmap import decode_heatmaps_batch, resize_coords

logger = logging.getLogger(__name__)

# Try importing wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Install with: pip install wandb")


class Trainer:
    """Trainer for keypoint detection model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: str = 'cuda'
    ):
        """Initialize trainer.
        
        Args:
            model: KPNet model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Training parameters
        self.epochs = config['epochs']
        self.learning_rate = float(config['learning_rate'])  # YAML에서 문자열로 읽힐 수 있으므로 float 변환
        self.weight_decay = float(config.get('weight_decay', 1e-4))  # YAML에서 문자열로 읽힐 수 있으므로 float 변환
        
        # Setup optimizer
        self.optimizer = self._get_optimizer()
        
        # Setup scheduler
        self.scheduler = self._get_scheduler()
        
        # Setup loss function
        self.criterion = self._get_criterion()
        
        # Checkpoint settings
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_best_only = config.get('save_best_only', True)
        self.best_val_loss = float('inf')
        
        # Logging
        self.log_frequency = config.get('log_frequency', 10)
        self.val_frequency = config.get('val_frequency', 1)
        
        # Metrics
        self.pck_threshold = config.get('pck_threshold', 10.0)
        self.heatmap_size = tuple(config['heatmap_size'])
        self.image_size = tuple(config['image_size'])
        
        # Wandb setup
        self.use_wandb = config.get('use_wandb', False) and WANDB_AVAILABLE
        if self.use_wandb:
            wandb_project = config.get('wandb_project', 'measure-extractor')
            wandb_entity = config.get('wandb_entity', None)
            try:
                # entity가 None이면 자동으로 사용자 계정 사용
                init_kwargs = {
                    'project': wandb_project,
                    'config': config,
                    'name': f"{wandb_project}-run",
                    'mode': 'online'  # online, offline, disabled
                }
                if wandb_entity:
                    init_kwargs['entity'] = wandb_entity
                
                wandb.init(**init_kwargs)
                logger.info(f"Wandb initialized: project={wandb_project}, entity={wandb_entity or 'auto'}")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}. Continuing without wandb logging.")
                self.use_wandb = False
        elif config.get('use_wandb', False) and not WANDB_AVAILABLE:
            logger.warning("wandb is enabled in config but not installed. Install with: pip install wandb")
        
        logger.info("Trainer initialized")
        logger.info(f"  Device: {device}")
        logger.info(f"  Epochs: {self.epochs}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Optimizer: {config.get('optimizer', 'adam')}")
    
    def _get_optimizer(self) -> optim.Optimizer:
        """Get optimizer."""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        return optimizer
    
    def _get_scheduler(self) -> Optional[Any]:
        """Get learning rate scheduler."""
        scheduler_name = self.config.get('scheduler', 'step').lower()
        
        if scheduler_name == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        elif scheduler_name == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs
            )
        elif scheduler_name == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.get('patience', 10),
                factor=self.config.get('gamma', 0.1)
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _get_criterion(self) -> nn.Module:
        """Get loss criterion."""
        loss_type = self.config.get('loss_type', 'mse').lower()
        
        if loss_type == 'mse':
            criterion = nn.MSELoss(reduction='none')
        elif loss_type == 'l1':
            criterion = nn.L1Loss(reduction='none')
        elif loss_type == 'smooth_l1':
            criterion = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        return criterion
    
    def _compute_loss(
        self,
        pred_heatmaps: torch.Tensor,
        gt_heatmaps: torch.Tensor,
        visibility: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss with visibility masking.
        
        Args:
            pred_heatmaps: Predicted heatmaps (B, K, H, W)
            gt_heatmaps: Ground truth heatmaps (B, K, H, W)
            visibility: Visibility mask (B, K)
            
        Returns:
            Scalar loss
        """
        # Compute per-pixel loss
        loss = self.criterion(pred_heatmaps, gt_heatmaps)  # (B, K, H, W)
        
        # Average over spatial dimensions
        loss = loss.mean(dim=[2, 3])  # (B, K)
        
        # Apply visibility mask
        if self.config.get('use_visibility_mask', True):
            loss = loss * visibility  # (B, K)
            
            # Average over visible keypoints
            num_visible = visibility.sum()
            if num_visible > 0:
                loss = loss.sum() / num_visible
            else:
                loss = loss.sum()
        else:
            loss = loss.mean()
        
        return loss
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            gt_heatmaps = batch['heatmaps'].to(self.device)
            visibility = batch['visibility'].to(self.device)
            
            # Forward pass
            pred_heatmaps = self.model(images)
            
            # Compute loss
            loss = self._compute_loss(pred_heatmaps, gt_heatmaps, visibility)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            if batch_idx % self.log_frequency == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        'train/batch_loss': loss.item(),
                        'train/epoch': epoch,
                        'train/batch': batch_idx
                    })
        
        avg_loss = total_loss / num_batches
        
        return {'loss': avg_loss}
    
    def evaluate(self, data_loader: DataLoader, desc: str = "Evaluating") -> Dict[str, float]:
        """Evaluate model on a data loader.
        
        Args:
            data_loader: DataLoader to evaluate on
            desc: Description for progress bar
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_pred_coords = []
        all_gt_coords = []
        all_visibility = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=desc):
                # Move to device
                images = batch['image'].to(self.device)
                gt_heatmaps = batch['heatmaps'].to(self.device)
                visibility = batch['visibility'].to(self.device)
                gt_keypoints = batch['keypoints'].numpy()  # (B, K, 2)
                
                # Forward pass
                pred_heatmaps = self.model(images)
                
                # Compute loss
                loss = self._compute_loss(pred_heatmaps, gt_heatmaps, visibility)
                total_loss += loss.item()
                
                # Decode heatmaps to coordinates
                pred_coords_heatmap = decode_heatmaps_batch(pred_heatmaps.cpu())  # (B, K, 2)
                
                # Resize coordinates to image resolution
                pred_coords_image = resize_coords(
                    pred_coords_heatmap,
                    from_size=(self.heatmap_size[1], self.heatmap_size[0]),
                    to_size=(self.image_size[1], self.image_size[0])
                )
                
                # Collect for metrics
                all_pred_coords.append(pred_coords_image)
                all_gt_coords.append(gt_keypoints)
                all_visibility.append(visibility.cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(data_loader)
        
        all_pred_coords = np.concatenate(all_pred_coords, axis=0)
        all_gt_coords = np.concatenate(all_gt_coords, axis=0)
        all_visibility = np.concatenate(all_visibility, axis=0)
        
        pck_metrics = compute_pck(all_pred_coords, all_gt_coords, all_visibility, self.pck_threshold)
        mse = compute_mse(all_pred_coords, all_gt_coords, all_visibility)
        
        metrics = {
            'loss': avg_loss,
            'pck': pck_metrics['pck'],
            'mean_distance': pck_metrics['mean_distance'],
            'mse': mse
        }
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate model on validation set.
        
        Returns:
            Dictionary with validation metrics
        """
        return self.evaluate(self.val_loader, desc="Validating")
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        if not self.save_best_only:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    def train(self):
        """Main training loop."""
        logger.info("=" * 60)
        logger.info("Starting training")
        logger.info("=" * 60)
        
        for epoch in range(1, self.epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch}/{self.epochs} - Train Loss: {train_metrics['loss']:.4f}")
            
            # Log train loss to wandb
            if self.use_wandb:
                wandb.log({
                    'train/epoch_loss': train_metrics['loss'],
                    'epoch': epoch
                })
            
            # Evaluate on train and validation sets
            val_metrics = None
            if epoch % self.val_frequency == 0:
                # Evaluate on train set
                train_eval_metrics = self.evaluate(self.train_loader, desc=f"Eval Train Epoch {epoch}")
                logger.info(f"Epoch {epoch}/{self.epochs} - Train Eval Loss: {train_eval_metrics['loss']:.4f}, "
                          f"PCK: {train_eval_metrics['pck']:.4f}, Mean Dist: {train_eval_metrics['mean_distance']:.2f}")
                
                # Evaluate on validation set
                val_metrics = self.validate()
                logger.info(f"Epoch {epoch}/{self.epochs} - Val Loss: {val_metrics['loss']:.4f}, "
                          f"PCK: {val_metrics['pck']:.4f}, Mean Dist: {val_metrics['mean_distance']:.2f}")
                
                # Log metrics to wandb
                if self.use_wandb:
                    log_dict = {
                        'train_eval/loss': train_eval_metrics['loss'],
                        'train_eval/pck': train_eval_metrics['pck'],
                        'train_eval/mean_distance': train_eval_metrics['mean_distance'],
                        'train_eval/mse': train_eval_metrics['mse'],
                        'val/loss': val_metrics['loss'],
                        'val/pck': val_metrics['pck'],
                        'val/mean_distance': val_metrics['mean_distance'],
                        'val/mse': val_metrics['mse'],
                        'epoch': epoch
                    }
                    wandb.log(log_dict)
                
                # Check if best model (based on validation loss)
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                    logger.info(f"New best validation loss: {self.best_val_loss:.4f}")
                    if self.use_wandb:
                        wandb.run.summary['best_val_loss'] = self.best_val_loss
                
                # Save checkpoint
                self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if val_metrics is not None:
                        self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Learning rate: {current_lr:.6f}")
                
                # Log learning rate to wandb
                if self.use_wandb:
                    wandb.log({
                        'learning_rate': current_lr,
                        'epoch': epoch
                    })
        
        logger.info("=" * 60)
        logger.info("Training completed")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("=" * 60)
        
        # Finish wandb run
        if self.use_wandb:
            wandb.finish()

