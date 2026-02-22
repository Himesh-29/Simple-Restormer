"""
Loss functions for image restoration.

Implements:
- CharbonnierLoss: Robust pixel-wise loss (sqrt((x-y)^2 + eps^2))
- FFTLoss: L1 loss in frequency domain via rfft2
- EdgeLoss: L1 loss on Laplacian-filtered images
- CombinedRestorationLoss: Weighted combination with NTIRE-validated lambdas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (pseudo-Huber): sqrt((pred - target)^2 + eps^2).
    
    More robust to outliers than L1, differentiable everywhere.
    Default eps=1e-3 follows common practice in restoration literature.
    """
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


class FFTLoss(nn.Module):
    """L1 loss in FFT frequency domain.
    
    Computes rfft2 of both pred and target, then measures L1 distance
    on real and imaginary components. Ensures spectral fidelity.
    """
    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm='backward')
        target_fft = torch.fft.rfft2(target, norm='backward')
        return F.l1_loss(
            torch.stack([pred_fft.real, pred_fft.imag], dim=-1),
            torch.stack([target_fft.real, target_fft.imag], dim=-1),
        )


class EdgeLoss(nn.Module):
    """Laplacian-based edge loss.
    
    Applies a 3x3 Laplacian kernel to extract edges from both pred and
    target, then computes L1 distance. Preserves high-frequency detail.
    """
    def __init__(self):
        super().__init__()
        # Laplacian kernel for edge detection
        kernel = torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
        self.register_buffer('kernel', kernel)

    def _laplacian(self, x):
        """Apply Laplacian per-channel."""
        b, c, h, w = x.shape
        # Expand kernel to match channels: (C, 1, 3, 3)
        kernel = self.kernel.expand(c, -1, -1, -1)
        return F.conv2d(x, kernel, padding=1, groups=c)

    def forward(self, pred, target):
        return F.l1_loss(self._laplacian(pred), self._laplacian(target))


class CombinedRestorationLoss(nn.Module):
    """Combined loss: Charbonnier + FFT + Edge.
    
    Default weights from NTIRE 2024 challenge winner analysis:
    - Charbonnier (λ=1.0): Primary pixel-wise reconstruction
    - FFT (λ=0.1): Spectral fidelity for frequency preservation
    - Edge (λ=0.05): High-frequency detail preservation via Laplacian
    """
    def __init__(self, charbonnier_weight=1.0, fft_weight=0.1,
                 edge_weight=0.05, eps=1e-3):
        super().__init__()
        self.charb = CharbonnierLoss(eps)
        self.fft = FFTLoss()
        self.edge = EdgeLoss()
        self.cw = charbonnier_weight
        self.fw = fft_weight
        self.ew = edge_weight

    def forward(self, pred, target):
        loss_charb = self.charb(pred, target)
        loss_fft = self.fft(pred, target)
        loss_edge = self.edge(pred, target)
        return self.cw * loss_charb + self.fw * loss_fft + self.ew * loss_edge
