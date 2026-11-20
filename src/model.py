"""
SeisNet-1D: Deep Learning Architecture
Implements a Multi-Task 1D Residual U-Net for simultaneous:
1. Event Detection (Binary Classification: Earthquake vs Noise)
2. Phase Picking (High-resolution P & S wave Gaussian probability curves over 6,000 samples)
"""

from typing import Tuple, Dict, Any, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    """
    Standard 1D Convolutional Block: Conv1d -> BatchNorm1d -> GELU
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1
    ):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResBlock1D(nn.Module):
    """
    1D Residual Block with optional strided downsampling:
    Path 1: Conv1d -> BatchNorm1d -> GELU -> Conv1d -> BatchNorm1d
    Path 2 (Skip): Identity or 1x1 Conv1d with stride -> BatchNorm1d
    Result: Act(Path1 + Path2)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 7
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act2(out + residual)
        return out


class DilatedBottleneck(nn.Module):
    """
    Bottleneck module with multi-scale dilated convolutions to expand the temporal
    receptive field across the 60-second seismic observation window.
    """
    def __init__(self, channels: int = 256, dilations: List[int] = [2, 4]):
        super().__init__()
        layers = []
        for d in dilations:
            layers.append(
                ConvBlock1D(channels, channels, kernel_size=3, dilation=d)
            )
        self.dilated_blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dilated_blocks(x)


class SeisNet1D(nn.Module):
    """
    SeisNet-1D: Multi-task 1D Residual U-Net Architecture.
    
    Inputs:
        Waveform tensor of shape (Batch, 3, 6000) representing [E, N, Z] components.
        
    Outputs:
        Tuple of (detection_logits, phase_logits):
        - detection_logits: (Batch,) scalar logit for earthquake probability (Earthquake vs Noise)
        - phase_logits: (Batch, 2, 6000) for [P-wave, S-wave] arrival probability curves.
    """
    def __init__(
        self,
        in_channels: int = 3,
        encoder_channels: List[int] = [32, 64, 128, 256],
        bottleneck_channels: int = 256,
        bottleneck_dilations: List[int] = [2, 4],
        dropout: float = 0.3,
        phase_classes: int = 2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.phase_classes = phase_classes

        # Stem: Initial feature extraction at native 6000 resolution
        stem_out = encoder_channels[0]
        self.stem = ConvBlock1D(in_channels, stem_out, kernel_size=7, stride=1, padding=3)

        # Encoder: 4 stages with exact integer downsamplings
        # 6000 -> 1500 (stride 4) -> 375 (stride 4) -> 75 (stride 5) -> 15 (stride 5)
        self.enc1 = ResBlock1D(stem_out, encoder_channels[1], stride=4, kernel_size=7)    # -> 1500
        self.enc2 = ResBlock1D(encoder_channels[1], encoder_channels[2], stride=4, kernel_size=7)  # -> 375
        self.enc3 = ResBlock1D(encoder_channels[2], encoder_channels[3], stride=5, kernel_size=7)  # -> 75
        self.enc4 = ResBlock1D(encoder_channels[3], bottleneck_channels, stride=5, kernel_size=7)  # -> 15

        # Bottleneck: Multi-scale dilated convs for 60s temporal context
        self.bottleneck = DilatedBottleneck(bottleneck_channels, dilations=bottleneck_dilations)

        # Head 1: Event Detection Head (GAP + GMP pooling -> MLP)
        self.global_pool_dim = bottleneck_channels * 2
        self.detection_head = nn.Sequential(
            nn.Linear(self.global_pool_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1)  # scalar logit (earthquake=1, noise=0)
        )

        # Head 2: Phase Picking Decoder (Symmetric Transposed Convolutions + Skip Connections)
        # UpStage 4: 15 -> 75 (stride 5)
        self.up4 = nn.ConvTranspose1d(bottleneck_channels, encoder_channels[3], kernel_size=5, stride=5)
        self.dec4 = ConvBlock1D(encoder_channels[3] + encoder_channels[3], encoder_channels[2], kernel_size=7, padding=3)

        # UpStage 3: 75 -> 375 (stride 5)
        self.up3 = nn.ConvTranspose1d(encoder_channels[2], encoder_channels[2], kernel_size=5, stride=5)
        self.dec3 = ConvBlock1D(encoder_channels[2] + encoder_channels[2], encoder_channels[1], kernel_size=7, padding=3)

        # UpStage 2: 375 -> 1500 (stride 4)
        self.up2 = nn.ConvTranspose1d(encoder_channels[1], encoder_channels[1], kernel_size=4, stride=4)
        self.dec2 = ConvBlock1D(encoder_channels[1] + encoder_channels[1], stem_out, kernel_size=7, padding=3)

        # UpStage 1: 1500 -> 6000 (stride 4)
        self.up1 = nn.ConvTranspose1d(stem_out, stem_out, kernel_size=4, stride=4)
        self.dec1 = ConvBlock1D(stem_out + stem_out, stem_out, kernel_size=7, padding=3)

        # Final 1x1 Convolution producing P and S logits
        self.phase_out = nn.Conv1d(stem_out, phase_classes, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Seismic waveform tensor (Batch, 3, 6000)
            
        Returns:
            Tuple:
            - detection_logits: Shape (Batch,)
            - phase_logits: Shape (Batch, 2, 6000)
        """
        # Encoder forward pass with skip connections
        s0 = self.stem(x)       # (B, 32, 6000)
        s1 = self.enc1(s0)      # (B, 64, 1500)
        s2 = self.enc2(s1)      # (B, 128, 375)
        s3 = self.enc3(s2)      # (B, 256, 75)
        s4 = self.enc4(s3)      # (B, 256, 15)

        # Bottleneck
        b = self.bottleneck(s4) # (B, 256, 15)

        # Detection Head: Concatenated Global Average & Global Max Pooling
        gap = torch.mean(b, dim=-1)   # (B, 256)
        gmp = torch.amax(b, dim=-1)   # (B, 256)
        pooled = torch.cat([gap, gmp], dim=-1) # (B, 512)
        detection_logits = self.detection_head(pooled).squeeze(-1) # (B,)

        # Phase Picking Decoder forward pass
        d4 = self.up4(b)                    # (B, 256, 75)
        d4 = torch.cat([d4, s3], dim=1)     # (B, 512, 75)
        d4 = self.dec4(d4)                  # (B, 128, 75)

        d3 = self.up3(d4)                   # (B, 128, 375)
        d3 = torch.cat([d3, s2], dim=1)     # (B, 256, 375)
        d3 = self.dec3(d3)                  # (B, 64, 375)

        d2 = self.up2(d3)                   # (B, 64, 1500)
        d2 = torch.cat([d2, s1], dim=1)     # (B, 128, 1500)
        d2 = self.dec2(d2)                  # (B, 32, 1500)

        d1 = self.up1(d2)                   # (B, 32, 6000)
        d1 = torch.cat([d1, s0], dim=1)     # (B, 64, 6000)
        d1 = self.dec1(d1)                  # (B, 32, 6000)

        phase_logits = self.phase_out(d1)   # (B, 2, 6000)

        return detection_logits, phase_logits


def get_model_summary(model: nn.Module, input_shape: Tuple[int, ...] = (1, 3, 6000)) -> Dict[str, Any]:
    """
    Computes and formats a comprehensive model parameter summary.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)

    summary = {
        "model_name": model.__class__.__name__,
        "input_shape": input_shape,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": round(model_size_mb, 2)
    }
    return summary
