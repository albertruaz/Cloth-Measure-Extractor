"""Keypoint Network (KPNet) for garment landmark detection.

Based on SimpleBaseline architecture:
ResNet backbone + Deconvolution head for heatmap regression.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import List
import logging

logger = logging.getLogger(__name__)


class KPNet(nn.Module):
    """Keypoint Network with ResNet backbone and deconvolution head."""
    
    def __init__(
        self,
        k_points: int,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        num_deconv_layers: int = 3,
        num_deconv_filters: List[int] = [256, 256, 256],
        num_deconv_kernels: List[int] = [4, 4, 4]
    ):
        """Initialize KPNet.
        
        Args:
            k_points: Number of keypoints to predict
            backbone: Backbone architecture ('resnet18', 'resnet34', 'resnet50', 'resnet101')
            pretrained: Whether to use pretrained backbone
            num_deconv_layers: Number of deconvolution layers
            num_deconv_filters: Number of filters for each deconv layer
            num_deconv_kernels: Kernel size for each deconv layer
        """
        super(KPNet, self).__init__()
        
        self.k_points = k_points
        self.num_deconv_layers = num_deconv_layers
        
        # Load backbone
        self.backbone = self._get_backbone(backbone, pretrained)
        
        # Get number of channels from backbone
        if backbone in ['resnet18', 'resnet34']:
            self.inplanes = 512
        else:  # resnet50, resnet101, resnet152
            self.inplanes = 2048
        
        # Build deconvolution head
        self.deconv_layers = self._make_deconv_layers(
            num_deconv_layers,
            num_deconv_filters,
            num_deconv_kernels
        )
        
        # Final layer to produce heatmaps
        self.final_layer = nn.Conv2d(
            in_channels=num_deconv_filters[-1],
            out_channels=k_points,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        logger.info(f"Created KPNet with {backbone} backbone")
        logger.info(f"  Keypoints: {k_points}")
        logger.info(f"  Deconv layers: {num_deconv_layers}")
        logger.info(f"  Pretrained: {pretrained}")
    
    def _get_backbone(self, backbone: str, pretrained: bool) -> nn.Module:
        """Get ResNet backbone."""
        if backbone == 'resnet18':
            weights = 'IMAGENET1K_V1' if pretrained else None
            resnet = models.resnet18(weights=weights)
        elif backbone == 'resnet34':
            weights = 'IMAGENET1K_V1' if pretrained else None
            resnet = models.resnet34(weights=weights)
        elif backbone == 'resnet50':
            weights = 'IMAGENET1K_V1' if pretrained else None
            resnet = models.resnet50(weights=weights)
        elif backbone == 'resnet101':
            weights = 'IMAGENET1K_V1' if pretrained else None
            resnet = models.resnet101(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove avgpool and fc layers
        backbone_modules = list(resnet.children())[:-2]
        backbone_model = nn.Sequential(*backbone_modules)
        
        return backbone_model
    
    def _make_deconv_layers(
        self,
        num_layers: int,
        num_filters: List[int],
        num_kernels: List[int]
    ) -> nn.Sequential:
        """Create deconvolution layers."""
        assert num_layers == len(num_filters), 'ERROR: num_layers != len(num_filters)'
        assert num_layers == len(num_kernels), 'ERROR: num_layers != len(num_kernels)'
        
        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            padding = 1
            output_padding = 0
            planes = num_filters[i]
            
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False
                )
            )
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Heatmaps of shape (B, K, H_out, W_out)
        """
        # Backbone
        x = self.backbone(x)  # (B, C, H/32, W/32)
        
        # Deconvolution head
        x = self.deconv_layers(x)  # (B, 256, H/4, W/4) with 3 deconv layers
        
        # Final heatmaps
        x = self.final_layer(x)  # (B, K, H/4, W/4)
        
        return x
    
    def init_weights(self):
        """Initialize weights for deconv layers."""
        for m in self.deconv_layers.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize final layer
        nn.init.normal_(self.final_layer.weight, std=0.001)
        nn.init.constant_(self.final_layer.bias, 0)


def create_model(config: dict) -> KPNet:
    """Create KPNet model from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        KPNet model
    """
    # Calculate number of keypoints
    names = config['names']
    k_points = 2 * len(names)  # Each measurement has 2 keypoints
    
    model = KPNet(
        k_points=k_points,
        backbone=config.get('backbone', 'resnet50'),
        pretrained=config.get('pretrained', True),
        num_deconv_layers=config.get('num_deconv_layers', 3),
        num_deconv_filters=config.get('num_deconv_filters', [256, 256, 256]),
        num_deconv_kernels=config.get('num_deconv_kernels', [4, 4, 4])
    )
    
    # Initialize weights
    model.init_weights()
    
    return model


def count_parameters(model: nn.Module) -> dict:
    """Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }

