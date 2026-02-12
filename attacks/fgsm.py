
import torch
import torch.nn as nn
from core import ATTACKERS, BaseAttacker, BaseModel
from typing import Any, Dict

@ATTACKERS.register_module("fgsm")
class FGSMAttacker(BaseAttacker):
    """Fast Gradient Sign Method (FGSM) Attacker."""
    
    def __init__(self, epsilon: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        
    def generate(self, model: BaseModel, images: torch.Tensor, targets: Any, **kwargs) -> torch.Tensor:
        """Generate FGSM adversarial examples.
        
        Args:
            model: The target model.
            images: Clean images (B, C, H, W). Normalized [0, 1].
            targets: Targets in format compatible with model.compute_loss().
            
        Returns:
            Adversarial images (B, C, H, W).
        """
        images = images.clone().detach().to(images.device)
        images.requires_grad = True
        
        # Forward pass
        preds = model.get_logits(images)
        
        # We need to compute loss. 
        # If model is Yolov8Wrapper, preds is (B, 4+nc, N).
        # compute_loss expects the output of forward(), which might be a tuple.
        # Yolov8Wrapper.get_logits returns preds[0] if it's a tuple.
        # But Yolov8Wrapper.compute_loss calls self.criterion(preds, targets).
        # Ultralytics loss expects the FULL tuple from forward().
        
        # So we should probably call model(images) directly instead of get_logits
        # if we want to use model.compute_loss.
        # BaseAttacker says "rely on get_logits".
        # But for detection, calculating loss from logits alone (without the other parts of the tuple)
        # might be tricky if the loss function expects the full tuple.
        
        # Let's try calling model(images) to get full preds for loss calculation.
        # This is allowed since we are inside the framework.
        preds_full = model(images)
        
        # Compute loss
        loss_dict = model.compute_loss(preds_full, targets)
        loss = loss_dict['loss']
        
        # Compute gradients
        model.zero_grad()
        loss.backward()
        
        # Generate adversarial images
        data_grad = images.grad.data
        sign_data_grad = data_grad.sign()
        
        # Perturb images
        # Untargeted attack: maximize loss -> x + epsilon * sign(grad)
        adv_images = images + self.epsilon * sign_data_grad
        
        # Clip to valid range [0, 1]
        adv_images = torch.clamp(adv_images, 0, 1)
        
        return adv_images.detach()
