"""Main training script for garment landmark regression."""
import os
import sys
import argparse
import logging
import torch
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, save_config
from src.utils.device import get_device
from src.models.kpnet import create_model, count_parameters
from src.data.dataset import create_data_loaders
from src.engine.trainer import Trainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(cfg_path: str = 'config.yaml'):
    """Main training function.
    
    Args:
        cfg_path: Path to configuration file
    """
    # Load configuration
    logger.info(f"Loading configuration from {cfg_path}")
    cfg = load_config(cfg_path)
    
    # Set random seed
    seed = cfg.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    logger.info(f"Random seed: {seed}")
    
    # Get device
    device_config = cfg.get('device', 'auto')
    device = get_device(device_config)
    
    # CUDA 최적화 설정
    if device.type == 'cuda':
        # cudnn.benchmark 설정 (입력 크기가 고정된 경우 성능 향상)
        if cfg.get('benchmark', True):
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmark enabled for faster training")
        else:
            torch.backends.cudnn.benchmark = False
        
        # 모든 CUDA 시드 설정 (재현성을 위해)
        torch.cuda.manual_seed_all(seed)
    
    # Create data loaders
    logger.info("=" * 60)
    logger.info("Creating data loaders")
    logger.info("=" * 60)
    
    train_loader, val_loader = create_data_loaders(
        train_csv=cfg['train_csv'],
        val_csv=cfg['val_csv'],
        config=cfg
    )
    
    # Create model
    logger.info("=" * 60)
    logger.info("Creating model")
    logger.info("=" * 60)
    
    model = create_model(cfg)
    
    # Print model info
    param_counts = count_parameters(model)
    logger.info(f"Model parameters:")
    logger.info(f"  Total: {param_counts['total']:,}")
    logger.info(f"  Trainable: {param_counts['trainable']:,}")
    logger.info(f"  Non-trainable: {param_counts['non_trainable']:,}")
    
    # Create trainer
    logger.info("=" * 60)
    logger.info("Creating trainer")
    logger.info("=" * 60)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=str(device)
    )
    
    # Save config to checkpoint directory
    checkpoint_dir = cfg.get('checkpoint_dir', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    config_save_path = os.path.join(checkpoint_dir, 'config.yaml')
    save_config(cfg, config_save_path)
    logger.info(f"Saved configuration to {config_save_path}")
    
    # Train
    trainer.train()
    
    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info(f"Best model saved to: {os.path.join(checkpoint_dir, 'best.pt')}")
    logger.info("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train garment landmark regression model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    
    args = parser.parse_args()
    
    try:
        main(args.config)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)

