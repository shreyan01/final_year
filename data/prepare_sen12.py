import os
import gdown
import zipfile
import shutil
from tqdm import tqdm
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

def download_sen12():
    """Download SEN1-2 dataset from Google Drive."""
    print("Downloading SEN1-2 dataset...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Download the dataset (SEN1-2 dataset from the official source)
    # Note: You need to request access to the dataset first at:
    # https://mediatum.ub.tum.de/1474000
    print("Please download the SEN1-2 dataset from:")
    print("https://mediatum.ub.tum.de/1474000")
    print("After downloading, place the zip file in data/raw/sen12.zip")
    
    # Wait for user to download and place the file
    input("Press Enter after you have placed the sen12.zip file in data/raw/")
    
    if not os.path.exists('data/raw/sen12.zip'):
        raise FileNotFoundError("sen12.zip not found in data/raw/. Please download it first.")
    
    # Extract the dataset
    print("Extracting dataset...")
    with zipfile.ZipFile('data/raw/sen12.zip', 'r') as zip_ref:
        zip_ref.extractall('data/raw')
    
    # Clean up
    os.remove('data/raw/sen12.zip')

def prepare_sen12():
    """Prepare SEN1-2 dataset for training."""
    print("Preparing SEN1-2 dataset...")
    
    # Create directories
    os.makedirs('data/train/sar', exist_ok=True)
    os.makedirs('data/train/optical', exist_ok=True)
    os.makedirs('data/train/ndvi', exist_ok=True)
    os.makedirs('data/train/water_mask', exist_ok=True)
    os.makedirs('data/train/urban_heatmap', exist_ok=True)
    
    # Process each scene
    scenes = os.listdir('data/raw/sen12')
    for scene in tqdm(scenes, desc="Processing scenes"):
        try:
            # Load SAR data (VV and VH channels)
            sar_path = os.path.join('data/raw/sen12', scene, 'sar.tif')
            if not os.path.exists(sar_path):
                print(f"Warning: SAR file not found for scene {scene}")
                continue
                
            with rasterio.open(sar_path) as src:
                sar_data = src.read()  # Shape: (2, H, W)
            
            # Load optical data (RGB channels)
            optical_path = os.path.join('data/raw/sen12', scene, 'optical.tif')
            if not os.path.exists(optical_path):
                print(f"Warning: Optical file not found for scene {scene}")
                continue
                
            with rasterio.open(optical_path) as src:
                optical_data = src.read()  # Shape: (3, H, W)
            
            # Calculate NDVI
            nir = optical_data[3]  # NIR band
            red = optical_data[0]  # Red band
            ndvi = (nir - red) / (nir + red + 1e-8)
            
            # Create water mask (simple threshold on VV channel)
            water_mask = (sar_data[0] < -15).astype(np.float32)
            
            # Create urban heatmap (simple threshold on VH channel)
            urban_heatmap = (sar_data[1] > -10).astype(np.float32)
            
            # Save processed data
            scene_name = scene.split('.')[0]
            
            # Save SAR data
            sar_output = os.path.join('data/train/sar', f'{scene_name}.tif')
            with rasterio.open(sar_output, 'w', **src.meta) as dst:
                dst.write(sar_data)
            
            # Save optical data
            optical_output = os.path.join('data/train/optical', f'{scene_name}.png')
            optical_rgb = np.transpose(optical_data[:3], (1, 2, 0))
            optical_rgb = ((optical_rgb - optical_rgb.min()) / (optical_rgb.max() - optical_rgb.min()) * 255).astype(np.uint8)
            Image.fromarray(optical_rgb).save(optical_output)
            
            # Save NDVI
            ndvi_output = os.path.join('data/train/ndvi', f'{scene_name}_ndvi.tif')
            with rasterio.open(ndvi_output, 'w', **src.meta) as dst:
                dst.write(ndvi[np.newaxis, ...])
            
            # Save water mask
            water_mask_output = os.path.join('data/train/water_mask', f'{scene_name}_water.tif')
            with rasterio.open(water_mask_output, 'w', **src.meta) as dst:
                dst.write(water_mask[np.newaxis, ...])
            
            # Save urban heatmap
            urban_heatmap_output = os.path.join('data/train/urban_heatmap', f'{scene_name}_urban.tif')
            with rasterio.open(urban_heatmap_output, 'w', **src.meta) as dst:
                dst.write(urban_heatmap[np.newaxis, ...])
                
        except Exception as e:
            print(f"Error processing scene {scene}: {str(e)}")
            continue

def verify_dataset():
    """Verify that the dataset is properly prepared."""
    print("\nVerifying dataset...")
    
    # Check if all required directories exist
    required_dirs = ['sar', 'optical', 'ndvi', 'water_mask', 'urban_heatmap']
    for dir_name in required_dirs:
        dir_path = os.path.join('data/train', dir_name)
        if not os.path.exists(dir_path):
            print(f"Error: Directory {dir_path} does not exist")
            return False
    
    # Count files in each directory
    for dir_name in required_dirs:
        dir_path = os.path.join('data/train', dir_name)
        files = os.listdir(dir_path)
        print(f"Found {len(files)} files in {dir_name}")
        
        if len(files) == 0:
            print(f"Error: No files found in {dir_path}")
            return False
    
    print("Dataset verification complete!")
    return True

if __name__ == '__main__':
    # Download and prepare dataset
    download_sen12()
    prepare_sen12()
    
    # Verify the dataset
    if verify_dataset():
        print("Dataset preparation complete!")
    else:
        print("Dataset preparation failed. Please check the errors above.") 