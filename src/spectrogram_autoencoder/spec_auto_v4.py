import torch
import torch.nn as nn

# ---------- 1-D UNet blocks ------------------------------------------------------
def down1d(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm1d(out_ch),
        nn.GELU(),
        nn.MaxPool1d(2)
    )

def up1d(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2),
        nn.GELU()
    )

class SpecAutoNet(nn.Module):
    """UNet that slices the spectrogram along time and
       learns frequency-only kernels with Conv1d."""
    def __init__(self, base_ch: int = 64):
        super().__init__()
        self.enc1 = down1d(1, base_ch)           # /2   →  64 ch
        self.enc2 = down1d(base_ch, 2*base_ch)   # /4   → 128 ch
        self.enc3 = down1d(2*base_ch, 4*base_ch) # /8   → 256 ch
        self.enc4 = down1d(4*base_ch, 8*base_ch) # /16  → 512 ch

        self.up4  = up1d(8*base_ch, 4*base_ch)   # /8   → 256
        self.up3  = up1d(4*base_ch, 2*base_ch)   # /4   → 128
        self.up2  = up1d(2*base_ch, base_ch)     # /2   → 64
        self.up1  = up1d(base_ch, base_ch // 2)  # /1   → 32

        self.dec4 = nn.Conv1d(8*base_ch, 4*base_ch, 3, padding=1)
        self.dec3 = nn.Conv1d(4*base_ch, 2*base_ch, 3, padding=1)
        self.dec2 = nn.Conv1d(2*base_ch,   base_ch, 3, padding=1)
        self.dec1 = nn.Conv1d(base_ch // 2, 1, 1)  # final 1-D conv

    def forward(self, x, return_latent: bool = False):
        """
        x : (B, 1, F, T)  – mono magnitude spectrogram
        All ops act along F; T is handled by reshaping.
        """
        B, C, F, T = x.shape
        if F % 16:
            raise RuntimeError(f"F={F} must be divisible by 16.")

        # (B, 1, F, T) → (B·T, 1, F)
        x = x.permute(0, 3, 1, 2).reshape(B*T, 1, F)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d4 = self.up4(e4)
        d3 = self.up3(self.dec4(torch.cat([d4, e3], 1)))
        d2 = self.up2(self.dec3(torch.cat([d3, e2], 1)))
        d1 = self.up1(self.dec2(torch.cat([d2, e1], 1)))
        d0 = self.dec1(d1)                 # (B·T, 1, F)

        # back to (B, 1, F, T)
        d0 = d0.reshape(B, T, 1, F).permute(0, 2, 3, 1).contiguous()
        return (d0, e4) if return_latent else d0
