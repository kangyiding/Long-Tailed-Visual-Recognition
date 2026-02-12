"""
Differentiable Preprocessing Utilities.
"""
import torch
import torch.nn.functional as F
from typing import Tuple

class DifferentiableLetterbox(torch.nn.Module):
    """
    Differentiable Letterbox padding and resizing.
    Maintains aspect ratio and pads with a specific value.
    """
    
    def __init__(self, target_size: Tuple[int, int], pad_value: float = 0.0, stride: int = 32):
        """
        Args:
            target_size (Tuple[int, int]): Target (H, W).
            pad_value (float): Padding value (normalized). 
                               Default 0.0 (for black padding) or 114/255 (gray).
            stride (int): Stride constraint (optional, to ensure output is divisible by stride).
        """
        super().__init__()
        self.target_h, self.target_w = target_size
        self.pad_value = pad_value
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Letterboxed tensor of shape (B, C, TargetH, TargetW).
        """
        B, C, H, W = x.shape
        
        # Calculate scale ratio
        r = min(self.target_h / H, self.target_w / W)
        
        # New dimensions
        new_h = int(round(H * r))
        new_w = int(round(W * r))
        
        # Resize (Differentiable)
        # mode='bilinear' is standard for images, align_corners=False usually matches cv2 better
        x_resized = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        # Padding
        dw = self.target_w - new_w
        dh = self.target_h - new_h
        
        # Divide padding by 2 to center
        pad_left = dw // 2
        pad_right = dw - pad_left
        pad_top = dh // 2
        pad_bottom = dh - pad_top
        
        # F.pad expects (pad_left, pad_right, pad_top, pad_bottom)
        # Note: F.pad constant value defaults to 0.
        x_padded = F.pad(x_resized, (pad_left, pad_right, pad_top, pad_bottom), value=self.pad_value)
        
        return x_padded
