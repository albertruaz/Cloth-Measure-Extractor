"""Device management utilities."""
import torch
import logging

logger = logging.getLogger(__name__)


def get_device(device_config: str = 'auto') -> torch.device:
    """Get torch device based on configuration.
    
    Args:
        device_config: Device configuration ('auto', 'cuda', 'cpu')
        
    Returns:
        torch.device object
    """
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)
    
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device

