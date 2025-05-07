import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import rasterio
from rasterio.windows import Window
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SARColorizationDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.root_dir = os.path.join(config['dataset'][split]['root_dir'])
        
        # Set up directories
        self.sar_dir = os.path.join(self.root_dir, config['dataset'][split]['sar_dir'])
        self.optical_dir = os.path.join(self.root_dir, config['dataset'][split]['optical_dir'])
        self.ndvi_dir = os.path.join(self.root_dir, config['dataset'][split]['ndvi_dir'])
        self.water_mask_dir = os.path.join(self.root_dir, config['dataset'][split]['water_mask_dir'])
        self.urban_heatmap_dir = os.path.join(self.root_dir, config['dataset'][split]['urban_heatmap_dir'])
        
        # Get image size
        self.image_size = config['dataset'][split]['image_size']
        
        # Get all image files
        self.sar_files = sorted([f for f in os.listdir(self.sar_dir) if f.endswith('.tif')])
        
        # Print dataset information
        print(f"\nDataset Information ({split}):")
        print(f"Root directory: {self.root_dir}")
        print(f"Number of SAR images: {len(self.sar_files)}")
        print(f"SAR directory: {self.sar_dir}")
        print(f"Optical directory: {self.optical_dir}")
        
        if not self.sar_files:
            raise ValueError(f"No .tif files found in {self.sar_dir}")
        
        # Set up transforms
        if split == 'train' and config['dataset'][split]['augment']:
            self.transform = A.Compose([
                A.RandomResizedCrop(height=self.image_size, width=self.image_size, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(p=1),
                    A.RandomGamma(p=1),
                ], p=0.3),
                A.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=self.image_size, width=self.image_size),
                A.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.sar_files)
    
    def __getitem__(self, idx):
        if idx >= len(self.sar_files):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.sar_files)} items")
            
        try:
            # Load SAR image
            sar_path = os.path.join(self.sar_dir, self.sar_files[idx])
            if not os.path.exists(sar_path):
                raise FileNotFoundError(f"SAR image not found: {sar_path}")
                
            with rasterio.open(sar_path) as src:
                sar_image = src.read()  # Shape: (C, H, W)
                sar_image = np.transpose(sar_image, (1, 2, 0))  # Shape: (H, W, C)
            
            # Load optical image
            optical_path = os.path.join(self.optical_dir, self.sar_files[idx].replace('.tif', '.png'))
            if not os.path.exists(optical_path):
                raise FileNotFoundError(f"Optical image not found: {optical_path}")
                
            optical_image = np.array(Image.open(optical_path))
            
            # Load NDVI if available
            ndvi = None
            ndvi_path = os.path.join(self.ndvi_dir, self.sar_files[idx].replace('.tif', '_ndvi.tif'))
            if os.path.exists(ndvi_path):
                with rasterio.open(ndvi_path) as src:
                    ndvi = src.read(1)  # Shape: (H, W)
            
            # Load water mask if available
            water_mask = None
            water_mask_path = os.path.join(self.water_mask_dir, self.sar_files[idx].replace('.tif', '_water.tif'))
            if os.path.exists(water_mask_path):
                with rasterio.open(water_mask_path) as src:
                    water_mask = src.read(1)  # Shape: (H, W)
            
            # Load urban heatmap if available
            urban_heatmap = None
            urban_heatmap_path = os.path.join(self.urban_heatmap_dir, self.sar_files[idx].replace('.tif', '_urban.tif'))
            if os.path.exists(urban_heatmap_path):
                with rasterio.open(urban_heatmap_path) as src:
                    urban_heatmap = src.read(1)  # Shape: (H, W)
            
            # Apply transforms
            transformed = self.transform(image=sar_image, mask=optical_image)
            sar_image = transformed['image']
            optical_image = transformed['mask']
            
            # Convert to tensor
            optical_image = torch.from_numpy(optical_image).permute(2, 0, 1).float() / 255.0
            
            # Process additional data if available
            if ndvi is not None:
                ndvi = torch.from_numpy(ndvi).float()
                ndvi = (ndvi - ndvi.min()) / (ndvi.max() - ndvi.min())
            
            if water_mask is not None:
                water_mask = torch.from_numpy(water_mask).float()
            
            if urban_heatmap is not None:
                urban_heatmap = torch.from_numpy(urban_heatmap).float()
                urban_heatmap = (urban_heatmap - urban_heatmap.min()) / (urban_heatmap.max() - urban_heatmap.min())
            
            # Create sample dictionary
            sample = {
                'sar': sar_image,
                'optical': optical_image,
                'filename': self.sar_files[idx]
            }
            
            if ndvi is not None:
                sample['ndvi'] = ndvi
            if water_mask is not None:
                sample['water_mask'] = water_mask
            if urban_heatmap is not None:
                sample['urban_heatmap'] = urban_heatmap
            
            return sample
            
        except Exception as e:
            print(f"Error loading item {idx} ({self.sar_files[idx]}): {str(e)}")
            raise

def get_dataloader(config, split='train'):
    dataset = SARColorizationDataset(config, split)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(split == 'train'),
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    return dataloader 