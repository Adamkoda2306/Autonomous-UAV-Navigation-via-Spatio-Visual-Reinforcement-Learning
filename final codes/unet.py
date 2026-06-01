"""
unet.py
-------
Multi-scale Encoder-Decoder U-Net Segmentation Network
Matches the architecture described in Section 3.2 of the paper:
  - Encoder: Conv2D(64) -> MaxPool -> Conv2D(128) -> MaxPool
  - Bottleneck: Conv2D(256)
  - Decoder: UpSampling + Skip-connections -> Conv2D(128) -> Conv2D(64)
  - Output: Conv1x1 + Sigmoid -> dense obstacle probability map [256x256x1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double Conv + ReLU block used in each encoder/decoder stage."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetSegmentation(nn.Module):
    """
    U-Net architecture as specified in Section 3.2.

    Input : I ∈ R^{256×256×3}   (monocular RGB)
    Output: M_prob ∈ R^{256×256×1}  (obstacle probability map, sigmoid output)

    Encoder path (equation 1-2 in paper):
        (256x256x3)  -> Conv -> (256x256x64)  -> MaxPool -> (128x128x64)
        (128x128x64) -> Conv -> (128x128x128) -> MaxPool -> (64x64x128)

    Bottleneck:
        (64x64x128) -> Conv -> (64x64x256)

    Decoder path (symmetric upsampling + skip connections):
        UpSample -> concat enc2 -> Conv -> (128x128x128)
        UpSample -> concat enc1 -> Conv -> (256x256x64)

    Output head:
        Conv1x1 + Sigmoid -> (256x256x1)
    """

    def __init__(self):
        super().__init__()

        # ---------- Encoder ----------
        self.enc1 = ConvBlock(3, 64)        # -> (256x256x64)
        self.pool1 = nn.MaxPool2d(2, 2)     # -> (128x128x64)

        self.enc2 = ConvBlock(64, 128)      # -> (128x128x128)
        self.pool2 = nn.MaxPool2d(2, 2)     # -> (64x64x128)

        # ---------- Bottleneck ----------
        self.bottleneck = ConvBlock(128, 256)   # -> (64x64x256)

        # ---------- Decoder ----------
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ConvBlock(256 + 128, 128)   # after concat with enc2 skip

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ConvBlock(128 + 64, 64)     # after concat with enc1 skip

        # ---------- Output head ----------
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)  # 1x1 projection
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: Tensor [B, 3, 256, 256]
        Returns: M_prob Tensor [B, 1, 256, 256]
        """
        # Encoder
        e1 = self.enc1(x)           # (B, 64, 256, 256)
        p1 = self.pool1(e1)         # (B, 64, 128, 128)

        e2 = self.enc2(p1)          # (B, 128, 128, 128)
        p2 = self.pool2(e2)         # (B, 128, 64, 64)

        # Bottleneck
        b = self.bottleneck(p2)     # (B, 256, 64, 64)

        # Decoder
        u2 = self.up2(b)            # (B, 256, 128, 128)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))  # (B, 128, 128, 128)

        u1 = self.up1(d2)           # (B, 128, 256, 256)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))  # (B, 64, 256, 256)

        # Output: M_prob = σ(Conv1x1(D_block2))
        out = self.sigmoid(self.out_conv(d1))   # (B, 1, 256, 256)
        return out


def load_unet(weights_path=None, device='cpu'):
    """
    Helper to instantiate U-Net and optionally load saved weights.

    Args:
        weights_path (str|None): Path to .pth checkpoint file.
        device (str): 'cpu' or 'cuda'.

    Returns:
        model (UNetSegmentation): ready-to-use model in eval mode.
    """
    model = UNetSegmentation().to(device)
    if weights_path is not None:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        print(f"[UNet] Loaded weights from {weights_path}")
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Quick sanity check
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetSegmentation().to(device)
    dummy = torch.zeros(1, 3, 256, 256).to(device)
    out = model(dummy)
    print(f"[UNet] Input shape : {dummy.shape}")
    print(f"[UNet] Output shape: {out.shape}")   # expected (1, 1, 256, 256)
    total = sum(p.numel() for p in model.parameters())
    print(f"[UNet] Total params: {total:,}")