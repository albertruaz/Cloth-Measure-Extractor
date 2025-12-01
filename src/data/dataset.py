"""Garment keypoint dataset for pants measurement."""
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
import io
import requests
import logging

try:
    from ..utils.utils import generate_gaussian_heatmap
except ImportError:
    from utils.utils import generate_gaussian_heatmap

logger = logging.getLogger(__name__)


class PantsMeasurementDataset(Dataset):
    """Pants Measurement Keypoint Dataset.
    
    Each measurement name has 2 keypoints (start and end).
    Loads from preprocessed CSV with columns: id, image_uri, category, measurements
    """
    
    def __init__(
        self,
        csv_path: str,
        names: List[str],
        image_size: Tuple[int, int],
        heatmap_size: Tuple[int, int],
        sigma: float,
        is_train: bool = True
    ):
        """Initialize dataset.
        
        Args:
            csv_path: Path to preprocessed CSV file (id, image_uri, category, measurements)
            names: List of measurement names (e.g., ['TOTAL_LENGTH', 'WAIST', ...])
            image_size: (height, width) for input images
            heatmap_size: (height, width) for output heatmaps
            sigma: Gaussian kernel sigma for heatmap generation
            is_train: Whether this is training set (for augmentation)
        """
        self.csv_path = csv_path
        self.names = names
        self.image_size = image_size  # (H, W)
        self.heatmap_size = heatmap_size  # (H, W)
        self.sigma = sigma
        self.is_train = is_train
        
        # Number of keypoints = 2 * number of measurement names
        self.num_keypoints = 2 * len(names)
        
        # Load CSV (탭으로 구분된 파일)
        self.df = pd.read_csv(csv_path, sep='\t')
        logger.info(f"Loaded {len(self.df)} samples from {csv_path}")
        
        # Create name to index mapping
        self.name_to_indices = {}
        for i, name in enumerate(names):
            self.name_to_indices[name] = (2 * i, 2 * i + 1)  # (start_idx, end_idx)
        
        # Setup transforms
        self.transform = self._get_transforms()
    
    def _get_transforms(self) -> A.Compose:
        """Get albumentations transforms.
        
        옷 사진에서 절대 좌우 반전/회전이 되면 안 되기 때문에
        학습/검증 모두 동일하게 Resize + Normalize만 적용한다.
        """
        transform = A.Compose(
            [
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ],
            keypoint_params=A.KeypointParams(
                format="xy",
                remove_invisible=False,
            ),
        )
        return transform
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load image from path or URL."""
        try:
            if image_path.startswith('http://') or image_path.startswith('https://'):
                # Load from URL
                response = requests.get(image_path, timeout=10)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert('RGB')
            else:
                # Load from local path
                full_path = image_path if not self.image_root else f"{self.image_root}/{image_path}"
                image = Image.open(full_path).convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            # Return a blank image as fallback
            return Image.new('RGB', (512, 512), color='gray')
    
    def _parse_measurements(self, request_body: str) -> Dict[str, Dict]:
        """Parse measurements from request_body JSON string.
        
        Expected format: {"name": "", "items": [{"name": "TOTAL_LENGTH", "x1": ..., ...}]}
        Returns: {"TOTAL_LENGTH": {"x1": ..., "y1": ..., ...}, ...}
        """
        try:
            data = json.loads(request_body)
            measurements = {}
            
            # 'items' 배열에서 measurement 추출
            if 'items' in data:
                for item in data['items']:
                    name = item.get('name')
                    if name:
                        measurements[name] = item
            # 또는 'measurements' 배열
            elif 'measurements' in data:
                for item in data['measurements']:
                    name = item.get('name')
                    if name:
                        measurements[name] = item
            
            return measurements
        except Exception as e:
            logger.error(f"Failed to parse request_body: {e}")
            return {}
    
    def _extract_keypoints(
        self,
        measurements: Dict[str, Dict],
        original_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract keypoints and visibility from measurements.
        
        Args:
            measurements: Dictionary of measurements {name: {x1, y1, x2, y2, ...}}
            original_size: (width, height) of original image
            
        Returns:
            keypoints: Array of shape (num_keypoints, 2) with (x, y) in original image coordinates
            visibility: Array of shape (num_keypoints,) - 1 if visible, 0 if not
        """
        keypoints = np.zeros((self.num_keypoints, 2), dtype=np.float32)
        visibility = np.zeros(self.num_keypoints, dtype=np.float32)
        
        orig_w, orig_h = original_size
        
        for name, measure_data in measurements.items():
            if name not in self.name_to_indices:
                continue
            
            x1 = measure_data.get('x1', 0)
            y1 = measure_data.get('y1', 0)
            x2 = measure_data.get('x2', 0)
            y2 = measure_data.get('y2', 0)
            
            # Check if measurement is missing (all zeros)
            if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                continue
            
            # Get indices for this measurement
            start_idx, end_idx = self.name_to_indices[name]
            
            # Store keypoints (in original image coordinates)
            keypoints[start_idx] = [x1, y1]
            keypoints[end_idx] = [x2, y2]
            
            # Mark as visible
            visibility[start_idx] = 1.0
            visibility[end_idx] = 1.0
        
        return keypoints, visibility
    
    def _generate_heatmaps(
        self,
        keypoints: np.ndarray,
        visibility: np.ndarray
    ) -> np.ndarray:
        """Generate Gaussian heatmaps for keypoints.
        
        Args:
            keypoints: Array of shape (num_keypoints, 2) with (x, y) in heatmap coordinates
            visibility: Array of shape (num_keypoints,)
            
        Returns:
            heatmaps: Array of shape (num_keypoints, H, W)
        """
        heatmaps = np.zeros((self.num_keypoints, self.heatmap_size[0], self.heatmap_size[1]), dtype=np.float32)
        
        for k in range(self.num_keypoints):
            if visibility[k] > 0:
                x, y = keypoints[k]
                heatmap = generate_gaussian_heatmap(self.heatmap_size, (x, y), self.sigma)
                heatmaps[k] = heatmap
        
        return heatmaps
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample.
        
        Returns:
            Dictionary with:
                - image: Tensor of shape (3, H, W)
                - heatmaps: Tensor of shape (K, H_hm, W_hm)
                - visibility: Tensor of shape (K,)
                - keypoints: Tensor of shape (K, 2) - for evaluation
        """
        row = self.df.iloc[idx]
        
        # Load image from URL
        image_uri = row['image_uri']
        image = self._load_image(image_uri)
        original_size = image.size  # (width, height)
        
        # Parse measurements from request_body
        request_body = row['request_body']
        measurements = self._parse_measurements(request_body)
        
        # Extract keypoints in original image coordinates
        keypoints_orig, visibility = self._extract_keypoints(measurements, original_size)
        
        # Convert to numpy array for albumentations
        image_np = np.array(image)
        
        # Prepare keypoints for albumentations (filter out invisible ones for transform)
        keypoints_list = []
        for k in range(self.num_keypoints):
            keypoints_list.append(tuple(keypoints_orig[k]))
        
        # Apply transforms
        transformed = self.transform(image=image_np, keypoints=keypoints_list)
        image_tensor = transformed['image']
        keypoints_transformed = np.array(transformed['keypoints'], dtype=np.float32)  # (K, 2)
        
        # Scale keypoints to heatmap resolution
        scale_x = self.heatmap_size[1] / self.image_size[1]
        scale_y = self.heatmap_size[0] / self.image_size[0]
        keypoints_heatmap = keypoints_transformed.copy()
        keypoints_heatmap[:, 0] *= scale_x
        keypoints_heatmap[:, 1] *= scale_y
        
        # Clip to heatmap bounds
        keypoints_heatmap[:, 0] = np.clip(keypoints_heatmap[:, 0], 0, self.heatmap_size[1] - 1)
        keypoints_heatmap[:, 1] = np.clip(keypoints_heatmap[:, 1], 0, self.heatmap_size[0] - 1)
        
        # Generate heatmaps
        heatmaps = self._generate_heatmaps(keypoints_heatmap, visibility)
        
        return {
            'image': image_tensor,
            'heatmaps': torch.from_numpy(heatmaps),
            'visibility': torch.from_numpy(visibility),
            'keypoints': torch.from_numpy(keypoints_transformed)  # For evaluation
        }


def create_data_loaders(
    train_csv: str,
    val_csv: str,
    config: Dict[str, Any]
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation data loaders.
    
    Args:
        train_csv: Path to training CSV
        val_csv: Path to validation CSV
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Extract config
    names = config['names']
    image_size = tuple(config['image_size'])
    heatmap_size = tuple(config['heatmap_size'])
    sigma = config['sigma']
    batch_size = config['batch_size']
    num_workers = config.get('num_workers', 4)
    pin_memory = config.get('pin_memory', True)
    
    # Create datasets
    train_dataset = PantsMeasurementDataset(
        train_csv, names, image_size, heatmap_size, sigma, is_train=True
    )
    val_dataset = PantsMeasurementDataset(
        val_csv, names, image_size, heatmap_size, sigma, is_train=False
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.info(f"Created data loaders:")
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    return train_loader, val_loader

