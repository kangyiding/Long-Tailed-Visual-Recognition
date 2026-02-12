
import torch
import torch.nn as nn
from core import ATTACKERS, BaseAttacker, BaseModel
from typing import Any, Dict

@ATTACKERS.register_module("pgd")
class PGDAttacker(BaseAttacker):
    """Projected Gradient Descent (PGD) Attacker."""
    
    def __init__(self, epsilon: float = 0.03, alpha: float = 0.01, steps: int = 10, random_start: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        
    def generate(self, model: BaseModel, images: torch.Tensor, targets: Any, **kwargs) -> torch.Tensor:
        """Generate PGD adversarial examples.
        
        Args:
            model: The target model.
            images: Clean images (B, C, H, W). Normalized [0, 1].
            targets: Targets.
            
        Returns:
            Adversarial images (B, C, H, W).
        """
        images = images.clone().detach().to(images.device)
        adv_images = images.clone().detach()
        
        if self.random_start:
            # Starting at a uniformly random point within the epsilon ball
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.epsilon, self.epsilon)
            adv_images = torch.clamp(adv_images, 0, 1)
            
        for step in range(self.steps):
            adv_images.requires_grad = True
            
            # Forward pass
            # We use full forward to ensure compatibility with loss computation
            preds = model(adv_images)
            
            loss_dict = model.compute_loss(preds, targets)
            loss = loss_dict['loss']
            
            model.zero_grad()
            loss.backward()
            
            data_grad = adv_images.grad.data
            
            # Update adversarial images
            adv_images = adv_images.detach() + self.alpha * data_grad.sign()
            
            # Project back to epsilon ball
            delta = torch.clamp(adv_images - images, -self.epsilon, self.epsilon)
            adv_images = torch.clamp(images + delta, 0, 1)
            
        return adv_images.detach()
