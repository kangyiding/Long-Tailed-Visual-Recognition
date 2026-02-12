
import torch
import torch.nn as nn
import numpy as np
from core import ATTACKERS, BaseAttacker, BaseModel
from typing import Any, Dict

@ATTACKERS.register_module("fog")
class FogAttacker(BaseAttacker):
    """Simulate Fog conditions."""
    
    def __init__(self, beta: float = 0.01, light: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.light = light
        
    def generate(self, model: BaseModel, images: torch.Tensor, targets: Any, **kwargs) -> torch.Tensor:
        """Apply Fog effect.
        
        This is currently a zero-order attack (simulation), not gradient-based optimization.
        """
        # images: (B, C, H, W) [0, 1]
        device = images.device
        B, C, H, W = images.shape
        
        # Simple uniform fog or depth-based if we had depth. 
        # Without depth, we can simulate 'thick fog' increasing with Y coordinate or just uniform.
        # Let's use a hybrid approach: random Perlin-like noise or simple gradient.
        
        # Transmission t(x) = exp(-beta * d(x))
        # Approximation: d(x) increases towards top of image (vanishing point)? 
        # Or just uniform t for global fog.
        
        t = torch.ones((B, 1, H, W), device=device) * np.exp(-self.beta * 10) # Constant fog
        
        # Atmospheric light A
        A = torch.ones_like(images) * self.light
        
        # I' = I*t + A*(1-t)
        adv_images = images * t + A * (1 - t)
        
        return torch.clamp(adv_images, 0, 1)

@ATTACKERS.register_module("rain")
class RainAttacker(BaseAttacker):
    """Simulate Rain conditions."""
    
    def __init__(self, density: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.density = density
        
    def generate(self, model: BaseModel, images: torch.Tensor, targets: Any, **kwargs) -> torch.Tensor:
        """Apply Rain effect (Streaks)."""
        device = images.device
        B, C, H, W = images.shape
        
        # Generate random noise
        noise = torch.rand((B, 1, H, W), device=device)
        
        # Threshold to create streaks
        rain_mask = (noise < self.density).float()
        
        # Blur/Streak effect (simplified)
        # In a real implementation, we would blur the noise in a direction.
        # Here we just add bright streaks.
        
        adv_images = images + rain_mask * 0.5 # Add streaks
        
        return torch.clamp(adv_images, 0, 1)
