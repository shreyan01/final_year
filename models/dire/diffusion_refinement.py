import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.activation = nn.SiLU()
        
    def forward(self, x, t):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)
        
        # Time embedding
        time_emb = self.time_mlp(t)
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)
        
        return h

class NDVIConditioner(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.conv(x)
        attention = self.attention(features)
        return features * attention

class WaterMaskRegularizer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv(x)
        return self.activation(x)

class UrbanHeatmapPrior(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.conv(x)
        attention = self.attention(features)
        return features * attention

class DiffusionRefinement(nn.Module):
    def __init__(self, 
                 in_channels=3,
                 time_dim=256,
                 base_channels=64):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels*2),
            nn.SiLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels*4),
            nn.SiLU()
        )
        
        # Physics-based guidance
        self.ndvi_conditioner = NDVIConditioner(base_channels*4, base_channels*4)
        self.water_mask = WaterMaskRegularizer(base_channels*4, base_channels*4)
        self.urban_heatmap = UrbanHeatmapPrior(base_channels*4, base_channels*4)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(base_channels*4, base_channels*4, time_dim)
            for _ in range(4)
        ])
        
        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, base_channels*2),
            nn.SiLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU()
        )
        
        # Output convolution
        self.output_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
        
    def forward(self, x, t, ndvi=None, water_mask=None, urban_heatmap=None):
        # Time embedding
        t = self.time_mlp(t)
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Downsampling
        x = self.down1(x)
        x = self.down2(x)
        
        # Physics-based guidance
        if ndvi is not None:
            x = x + self.ndvi_conditioner(ndvi)
        if water_mask is not None:
            x = x * self.water_mask(water_mask)
        if urban_heatmap is not None:
            x = x + self.urban_heatmap(urban_heatmap)
        
        # Residual blocks
        for res_block in self.res_blocks:
            x = x + res_block(x, t)
        
        # Upsampling
        x = self.up1(x)
        x = self.up2(x)
        
        # Output
        x = self.output_conv(x)
        return x

class DiffusionProcess:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        alphas_cumprod_t = self.alphas_cumprod[t]
        return torch.sqrt(alphas_cumprod_t) * x + torch.sqrt(1 - alphas_cumprod_t) * noise, noise
    
    def remove_noise(self, model, x, t, ndvi=None, water_mask=None, urban_heatmap=None):
        with torch.no_grad():
            predicted_noise = model(x, t, ndvi, water_mask, urban_heatmap)
            alphas_cumprod_t = self.alphas_cumprod[t]
            x0 = (x - torch.sqrt(1 - alphas_cumprod_t) * predicted_noise) / torch.sqrt(alphas_cumprod_t)
            return x0

if __name__ == "__main__":
    # Test the model
    model = DiffusionRefinement()
    diffusion = DiffusionProcess()
    
    # Test input
    x = torch.randn(2, 3, 256, 256)  # Batch of 2, 3 channels, 256x256 images
    t = torch.randint(0, 1000, (2,))  # Random timesteps
    
    # Test forward pass
    output = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test diffusion process
    noisy_x, noise = diffusion.add_noise(x, t)
    denoised_x = diffusion.remove_noise(model, noisy_x, t)
    print(f"Noisy input shape: {noisy_x.shape}")
    print(f"Denoised output shape: {denoised_x.shape}") 