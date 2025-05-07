import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
from tqdm import tqdm
import wandb

from models.sat.speckle_aware_transformer import SpeckleAwareTransformer
from models.ch_gan.color_hallucination_gan import Generator, Discriminator
from models.dire.diffusion_refinement import DiffusionRefinement, DiffusionProcess

class HNDPGAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sat = SpeckleAwareTransformer(
            in_channels=config['sat']['in_channels'],
            dim=config['sat']['dim'],
            depth=config['sat']['depth'],
            num_heads=config['sat']['num_heads']
        )
        
        self.ch_gan = Generator(
            in_channels=config['ch_gan']['in_channels'],
            out_channels=config['ch_gan']['out_channels']
        )
        
        self.dire = DiffusionRefinement(
            in_channels=config['dire']['in_channels'],
            time_dim=config['dire']['time_dim'],
            base_channels=config['dire']['base_channels']
        )
        
        self.diffusion = DiffusionProcess(
            num_timesteps=config['dire']['num_timesteps'],
            beta_start=config['dire']['beta_start'],
            beta_end=config['dire']['beta_end']
        )
        
    def forward(self, x, t=None, ndvi=None, water_mask=None, urban_heatmap=None):
        # Stage 1: Speckle-Aware Transformer
        x = self.sat(x)
        
        # Stage 2: Color Hallucination GAN
        x = self.ch_gan(x)
        
        # Stage 3: Diffusion-Driven Refinement
        if t is not None:
            x = self.dire(x, t, ndvi, water_mask, urban_heatmap)
        
        return x

class TripleAdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, real_pred, fake_pred, halluc_pred):
        real_loss = self.bce(real_pred, torch.ones_like(real_pred))
        fake_loss = self.bce(fake_pred, torch.zeros_like(fake_pred))
        halluc_loss = self.bce(halluc_pred, torch.zeros_like(halluc_pred))
        return real_loss + fake_loss + halluc_loss

class TerrainConsistentLoss(nn.Module):
    def __init__(self, vgg_model):
        super().__init__()
        self.vgg = vgg_model
        self.criterion = nn.MSELoss()
        
    def forward(self, output, target):
        output_features = self.vgg(output)
        target_features = self.vgg(target)
        return self.criterion(output_features, target_features)

def train(config):
    # Initialize wandb
    wandb.init(project="sar-colorization", config=config)
    
    # Create models
    model = HNDPGAModel(config).to(config['device'])
    discriminator = Discriminator().to(config['device'])
    
    # Create optimizers
    model_optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=config['training']['learning_rate'])
    
    # Create loss functions
    adversarial_loss = TripleAdversarialLoss().to(config['device'])
    terrain_loss = TerrainConsistentLoss(config['vgg_model']).to(config['device'])
    
    # Create data loaders
    train_loader = DataLoader(
        config['dataset']['train'],
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    
    # Training loop
    for epoch in range(config['training']['num_epochs']):
        model.train()
        discriminator.train()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        for batch in progress_bar:
            # Get data
            sar_images = batch['sar'].to(config['device'])
            optical_images = batch['optical'].to(config['device'])
            ndvi = batch.get('ndvi', None)
            water_mask = batch.get('water_mask', None)
            urban_heatmap = batch.get('urban_heatmap', None)
            
            # Generate timesteps for diffusion
            t = torch.randint(0, config['dire']['num_timesteps'], (sar_images.shape[0],)).to(config['device'])
            
            # Train discriminator
            disc_optimizer.zero_grad()
            
            # Generate fake images
            fake_images = model(sar_images, t, ndvi, water_mask, urban_heatmap)
            
            # Get discriminator predictions
            real_pred = discriminator(optical_images)
            fake_pred = discriminator(fake_images.detach())
            halluc_pred = discriminator(model(sar_images, None, ndvi, water_mask, urban_heatmap).detach())
            
            # Calculate discriminator loss
            disc_loss = adversarial_loss(real_pred, fake_pred, halluc_pred)
            disc_loss.backward()
            disc_optimizer.step()
            
            # Train generator
            model_optimizer.zero_grad()
            
            # Get discriminator predictions for generator training
            fake_pred = discriminator(fake_images)
            
            # Calculate generator losses
            gen_adv_loss = adversarial_loss(real_pred, fake_pred, None)
            gen_terrain_loss = terrain_loss(fake_images, optical_images)
            
            # Total generator loss
            gen_loss = gen_adv_loss + config['training']['terrain_loss_weight'] * gen_terrain_loss
            gen_loss.backward()
            model_optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix({
                'disc_loss': disc_loss.item(),
                'gen_loss': gen_loss.item()
            })
            
            # Log to wandb
            wandb.log({
                'disc_loss': disc_loss.item(),
                'gen_loss': gen_loss.item(),
                'gen_adv_loss': gen_adv_loss.item(),
                'gen_terrain_loss': gen_terrain_loss.item()
            })
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_interval'] == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'model_optimizer_state_dict': model_optimizer.state_dict(),
                'disc_optimizer_state_dict': disc_optimizer.state_dict()
            }
            torch.save(checkpoint, os.path.join(config['training']['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pt'))

if __name__ == "__main__":
    # Load configuration
    with open('configs/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create checkpoint directory
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    
    # Start training
    train(config) 