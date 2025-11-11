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
    elif device_config == 'cuda':
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device(device_config)
    
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"Current GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        # Set default CUDA device
        torch.cuda.set_device(0)
    
    return device

