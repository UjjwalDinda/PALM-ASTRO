
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    """
    Lightweight UNet. Names match the app (enc1, enc2, enc3, dec3, dec2, final)
    Output: raw logits with shape [B, n_classes, H, W]
    """
    def __init__(self, n_classes: int = 4):
        super().__init__()
        # encoder
        self.enc1 = DoubleConv(3, 32)    # -> 32 ch
        self.enc2 = DoubleConv(32, 64)   # -> 64 ch
        self.enc3 = DoubleConv(64, 128)  # -> 128 ch

        self.pool = nn.MaxPool2d(2)

        # decoder (note: inputs after concat)
        # up conv uses ConvTranspose2d
        self.up = nn.ConvTranspose2d

        # dec3: up(enc3) => 64, concat enc2 (64) => 128 -> reduce to 64
        self.dec3 = DoubleConv(128, 64)
        # dec2: up(dec3) => 32, concat enc1 (32) => 64 -> reduce to 32
        self.dec2 = DoubleConv(64, 32)

        self.final = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # encoder
        e1 = self.enc1(x)               # [B,32,H,W]
        e2 = self.enc2(self.pool(e1))   # [B,64,H/2,W/2]
        e3 = self.enc3(self.pool(e2))   # [B,128,H/4,W/4]

        # decoder
        d3 = self.up(128, 64, kernel_size=2, stride=2)(e3)  # [B,64,H/2,W/2]
        d3 = torch.cat([d3, e2], dim=1)                    # [B,128,H/2,W/2]
        d3 = self.dec3(d3)                                 # [B,64,H/2,W/2]

        d2 = self.up(64, 32, kernel_size=2, stride=2)(d3)  # [B,32,H,W]
        d2 = torch.cat([d2, e1], dim=1)                    # [B,64,H,W]
        d2 = self.dec2(d2)                                 # [B,32,H,W]

        out = self.final(d2)                               # [B,n_classes,H,W]
        return out
