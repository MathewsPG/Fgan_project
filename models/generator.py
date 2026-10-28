import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM) adaptively rescales feature channels 
    by modeling inter-channel dependencies (rain streaks vs clean semantics).
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        reduced_planes = max(in_planes // ratio, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, reduced_planes, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM) highlights dense haze clusters and rain streaks
    spatially across road scenes (skyline, road surface, vehicles).
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(concat)
        return self.sigmoid(out)


class DualAttentionBlock(nn.Module):
    """
    Combines Channel and Spatial Attention sequentially to guide feature refinement.
    """
    def __init__(self, in_channels):
        super(DualAttentionBlock, self).__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ResidualAttentionBlock(nn.Module):
    """
    Residual Block enhanced with Dual Attention for fine-grained texture preservation.
    """
    def __init__(self, dim):
        super(ResidualAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(dim)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(dim)
        self.attention = DualAttentionBlock(dim)

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.attention(out)
        return residual + out


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, down=True, use_act=True, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_c, out_c, **kwargs),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)


class DeRainDeHazeGenerator(nn.Module):
    """
    Feature-Attention U-Net Generator with Global Residual Learning:
    Given degraded road image I (rain + haze), outputs clean road image J.
    Uses multi-scale encoder-decoder with skip connections, bottleneck residual
    attention blocks, and residual output formulation: J = I + Delta_I.
    """
    def __init__(self, in_channels=3, out_channels=3, features=64, num_residual_blocks=6):
        super(DeRainDeHazeGenerator, self).__init__()
        
        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(features),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Downsampling Encoder
        self.down1 = ConvBlock(features, features * 2, down=True, kernel_size=4, stride=2, padding=1)
        self.down2 = ConvBlock(features * 2, features * 4, down=True, kernel_size=4, stride=2, padding=1)
        self.down3 = ConvBlock(features * 4, features * 8, down=True, kernel_size=4, stride=2, padding=1)
        
        # Bottleneck: Multi-stage Dual Attention Residual Blocks
        rab_layers = [ResidualAttentionBlock(features * 8) for _ in range(num_residual_blocks)]
        self.bottleneck = nn.Sequential(*rab_layers)
        
        # Upsampling Decoder with Skip Connections
        self.up1 = ConvBlock(features * 8, features * 4, down=False, kernel_size=4, stride=2, padding=1)
        self.up2 = ConvBlock(features * 8, features * 2, down=False, kernel_size=4, stride=2, padding=1)
        self.up3 = ConvBlock(features * 4, features, down=False, kernel_size=4, stride=2, padding=1)
        
        # Refinement & Output head
        self.refine = nn.Sequential(
            ResidualAttentionBlock(features * 2),
            nn.Conv2d(features * 2, features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, out_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.Tanh()
        )

    def forward(self, x):
        # Initial
        d0 = self.initial(x)
        # Encoder
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        
        # Bottleneck with Attention
        bn = self.bottleneck(d3)
        
        # Decoder with U-Net skips
        u1 = self.up1(bn)
        u1 = torch.cat([u1, d2], dim=1)
        
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d0], dim=1)
        
        # Predict residual correction map
        residual = self.refine(u3)
        
        # Final output with residual skip: input + residual mapped back to [-1, 1]
        out = torch.clamp(x + residual, -1.0, 1.0)
        return out


if __name__ == "__main__":
    model = DeRainDeHazeGenerator(in_channels=3, out_channels=3)
    dummy_input = torch.randn(1, 3, 256, 256)
    out = model(dummy_input)
    print("Generator forward pass successful. Output shape:", out.shape)
