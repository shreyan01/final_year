import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super().__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self._make_params()

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1))
        v = nn.Parameter(w.data.new(width).normal_(0, 1))
        u.requires_grad = False
        v.requires_grad = False
        self.module.register_buffer('u', u)
        self.module.register_buffer('v', v)

    def _power_method(self, w, eps=1e-12):
        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v = F.normalize(torch.mv(w.view(height, -1).t(), self.module.u), dim=0, eps=eps)
            u = F.normalize(torch.mv(w.view(height, -1), v), dim=0, eps=eps)
        sigma = u.dot(w.view(height, -1).mv(v))
        return u, v, sigma

    def forward(self, *args):
        w = getattr(self.module, self.name)
        u, v, sigma = self._power_method(w)
        self.module.u.copy_(u)
        self.module.v.copy_(v)
        w_sn = w / sigma
        setattr(self.module, self.name, w_sn)
        return self.module(*args)

class ChrominanceWarping(nn.Module):
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

class BandwidthExpansion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x

class Generator(nn.Module):
    def __init__(self, in_channels=2, out_channels=3):
        super().__init__()
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.InstanceNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.InstanceNorm2d(256)
            ) for _ in range(9)]
        )
        
        # Chrominance warping
        self.chrom_warp = ChrominanceWarping(256, 256)
        
        # Bandwidth expansion
        self.band_exp = BandwidthExpansion(256, 256)
        
        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Output convolution
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=7, padding=3)
        
    def forward(self, x):
        # Initial convolution
        x = self.init_conv(x)
        
        # Downsampling
        x = self.down1(x)
        x = self.down2(x)
        
        # Residual blocks
        identity = x
        x = self.res_blocks(x)
        x = x + identity
        
        # Chrominance warping
        x = self.chrom_warp(x)
        
        # Bandwidth expansion
        x = self.band_exp(x)
        
        # Upsampling
        x = self.up1(x)
        x = self.up2(x)
        
        # Output
        x = self.output_conv(x)
        return torch.tanh(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )
        
        # Apply spectral normalization
        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.Conv2d):
                self.model[i] = SpectralNorm(layer)
        
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Test the models
    generator = Generator()
    discriminator = Discriminator()
    
    # Test input
    x = torch.randn(2, 2, 256, 256)  # Batch of 2, 2 channels (SAR), 256x256 images
    
    # Test generator
    gen_output = generator(x)
    print(f"Generator input shape: {x.shape}")
    print(f"Generator output shape: {gen_output.shape}")
    
    # Test discriminator
    disc_output = discriminator(gen_output)
    print(f"Discriminator output shape: {disc_output.shape}") 