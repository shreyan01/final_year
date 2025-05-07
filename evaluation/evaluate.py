import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import yaml
from PIL import Image
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from pytorch_fid import fid_score
import matplotlib.pyplot as plt

from models.sat.speckle_aware_transformer import SpeckleAwareTransformer
from models.ch_gan.color_hallucination_gan import Generator, Discriminator
from models.dire.diffusion_refinement import DiffusionRefinement, DiffusionProcess
from data.dataset import get_dataloader

class SARColorScore:
    def __init__(self):
        self.lpips_fn = lpips.LPIPS(net='alex')
        
    def spectral_angle_mapper(self, img1, img2):
        """Calculate Spectral Angle Mapper between two images."""
        img1_norm = img1 / (np.linalg.norm(img1, axis=2, keepdims=True) + 1e-8)
        img2_norm = img2 / (np.linalg.norm(img2, axis=2, keepdims=True) + 1e-8)
        dot_product = np.sum(img1_norm * img2_norm, axis=2)
        return np.mean(np.arccos(np.clip(dot_product, -1.0, 1.0)))
    
    def color_histogram_matching(self, img1, img2, bins=64):
        """Calculate color histogram matching score."""
        hist1 = np.histogram(img1.reshape(-1, 3), bins=bins, range=(0, 1))[0]
        hist2 = np.histogram(img2.reshape(-1, 3), bins=bins, range=(0, 1))[0]
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        return np.sum(np.minimum(hist1, hist2))
    
    def calculate_score(self, generated, target):
        """Calculate SARColorScore combining SAM, LPIPS, and color histogram matching."""
        # Convert to numpy for SAM and histogram matching
        gen_np = generated.cpu().numpy().transpose(0, 2, 3, 1)
        target_np = target.cpu().numpy().transpose(0, 2, 3, 1)
        
        # Calculate SAM
        sam_scores = [self.spectral_angle_mapper(g, t) for g, t in zip(gen_np, target_np)]
        sam_score = np.mean(sam_scores)
        
        # Calculate LPIPS
        lpips_score = self.lpips_fn(generated, target).mean().item()
        
        # Calculate color histogram matching
        hist_scores = [self.color_histogram_matching(g, t) for g, t in zip(gen_np, target_np)]
        hist_score = np.mean(hist_scores)
        
        # Combine scores (normalized to [0, 1])
        final_score = (1 - sam_score/np.pi) * 0.4 + (1 - lpips_score) * 0.4 + hist_score * 0.2
        return final_score

def evaluate(config):
    # Load model
    model = torch.load(config['evaluation']['model_path'])
    model.eval()
    
    # Create dataloader
    test_loader = get_dataloader(config, split='test')
    
    # Initialize metrics
    metrics = {
        'psnr': [],
        'ssim': [],
        'lpips': [],
        'sar_color_score': []
    }
    
    # Initialize SARColorScore
    sar_color_scorer = SARColorScore()
    
    # Create save directory
    os.makedirs(config['evaluation']['save_dir'], exist_ok=True)
    
    # Evaluation loop
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Get data
            sar_images = batch['sar'].to(config['device'])
            optical_images = batch['optical'].to(config['device'])
            ndvi = batch.get('ndvi', None)
            water_mask = batch.get('water_mask', None)
            urban_heatmap = batch.get('urban_heatmap', None)
            
            # Generate images
            generated_images = model(sar_images, None, ndvi, water_mask, urban_heatmap)
            
            # Calculate metrics
            for i in range(generated_images.shape[0]):
                gen_img = generated_images[i].cpu().numpy().transpose(1, 2, 0)
                target_img = optical_images[i].cpu().numpy().transpose(1, 2, 0)
                
                # PSNR
                metrics['psnr'].append(psnr(target_img, gen_img))
                
                # SSIM
                metrics['ssim'].append(ssim(target_img, gen_img, multichannel=True))
                
                # LPIPS
                metrics['lpips'].append(
                    lpips.LPIPS(net='alex')(
                        generated_images[i:i+1],
                        optical_images[i:i+1]
                    ).item()
                )
                
                # SARColorScore
                metrics['sar_color_score'].append(
                    sar_color_scorer.calculate_score(
                        generated_images[i:i+1],
                        optical_images[i:i+1]
                    )
                )
            
            # Save sample images
            if len(metrics['psnr']) <= config['evaluation']['num_samples']:
                for i in range(generated_images.shape[0]):
                    # Save SAR image
                    sar_img = sar_images[i].cpu().numpy().transpose(1, 2, 0)
                    sar_img = (sar_img - sar_img.min()) / (sar_img.max() - sar_img.min())
                    plt.imsave(
                        os.path.join(config['evaluation']['save_dir'], f'sar_{len(metrics["psnr"])}.png'),
                        sar_img
                    )
                    
                    # Save generated image
                    gen_img = generated_images[i].cpu().numpy().transpose(1, 2, 0)
                    gen_img = (gen_img - gen_img.min()) / (gen_img.max() - gen_img.min())
                    plt.imsave(
                        os.path.join(config['evaluation']['save_dir'], f'generated_{len(metrics["psnr"])}.png'),
                        gen_img
                    )
                    
                    # Save target image
                    target_img = optical_images[i].cpu().numpy().transpose(1, 2, 0)
                    plt.imsave(
                        os.path.join(config['evaluation']['save_dir'], f'target_{len(metrics["psnr"])}.png'),
                        target_img
                    )
    
    # Calculate FID score
    fid_score_value = fid_score.calculate_fid_given_paths(
        [config['evaluation']['save_dir']],
        [config['evaluation']['test_dir']],
        batch_size=config['training']['batch_size'],
        device=config['device'],
        dims=2048
    )
    
    # Print results
    print("\nEvaluation Results:")
    print(f"PSNR: {np.mean(metrics['psnr']):.2f} ± {np.std(metrics['psnr']):.2f}")
    print(f"SSIM: {np.mean(metrics['ssim']):.4f} ± {np.std(metrics['ssim']):.4f}")
    print(f"LPIPS: {np.mean(metrics['lpips']):.4f} ± {np.std(metrics['lpips']):.4f}")
    print(f"SARColorScore: {np.mean(metrics['sar_color_score']):.4f} ± {np.std(metrics['sar_color_score']):.4f}")
    print(f"FID: {fid_score_value:.2f}")
    
    # Save results to file
    results = {
        'psnr': float(np.mean(metrics['psnr'])),
        'psnr_std': float(np.std(metrics['psnr'])),
        'ssim': float(np.mean(metrics['ssim'])),
        'ssim_std': float(np.std(metrics['ssim'])),
        'lpips': float(np.mean(metrics['lpips'])),
        'lpips_std': float(np.std(metrics['lpips'])),
        'sar_color_score': float(np.mean(metrics['sar_color_score'])),
        'sar_color_score_std': float(np.std(metrics['sar_color_score'])),
        'fid': float(fid_score_value)
    }
    
    with open(os.path.join(config['evaluation']['save_dir'], 'results.yaml'), 'w') as f:
        yaml.dump(results, f)

if __name__ == "__main__":
    # Load configuration
    with open('configs/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Add model path to config
    config['evaluation']['model_path'] = 'checkpoints/best_model.pt'
    
    # Run evaluation
    evaluate(config) 