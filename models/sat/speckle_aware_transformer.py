import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PolarimetricAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ScatteringMechanismEmbedding(nn.Module):
    def __init__(self, dim, num_mechanisms=3):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(1, num_mechanisms, dim))
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B = x.shape[0]
        embeddings = repeat(self.embeddings, '1 n d -> b n d', b=B)
        return self.proj(embeddings)

class MultiplicativeNoiseLayer(nn.Module):
    def __init__(self, alpha=0.9):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x):
        noise = torch.sqrt(x) * torch.randn_like(x)
        return x + (1 - self.alpha) * noise

class SpeckleAwareTransformer(nn.Module):
    def __init__(self, 
                 in_channels=2,  # VV/VH channels
                 dim=256,
                 depth=12,
                 num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.1,
                 attn_drop_rate=0.1):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=7, stride=2, padding=3),
            nn.LayerNorm([dim, 64, 64])  # Assuming 128x128 input
        )
        
        # Scattering mechanism embeddings
        self.scattering_emb = ScatteringMechanismEmbedding(dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                PolarimetricAttention(dim, num_heads, qkv_bias, attn_drop_rate, drop_rate),
                nn.LayerNorm(dim),
                nn.Linear(dim, int(dim * mlp_ratio)),
                nn.GELU(),
                nn.Linear(int(dim * mlp_ratio), dim),
                nn.Dropout(drop_rate)
            ) for _ in range(depth)
        ])
        
        # Multiplicative noise layer
        self.noise_layer = MultiplicativeNoiseLayer()
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.ConvTranspose2d(dim, in_channels, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        B, C, H, W = x.shape
        
        # Reshape for transformer
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Add scattering mechanism embeddings
        scattering_emb = self.scattering_emb(x)
        x = x + scattering_emb
        
        # Apply transformer blocks
        for block in self.blocks:
            x = x + block(x)
            
        # Reshape back to image
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        
        # Apply multiplicative noise
        x = self.noise_layer(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return x

if __name__ == "__main__":
    # Test the model
    model = SpeckleAwareTransformer()
    x = torch.randn(2, 2, 128, 128)  # Batch of 2, 2 channels (VV/VH), 128x128 images
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
