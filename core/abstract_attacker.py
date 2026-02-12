from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch

from core.abstract_model import BaseModel

class BaseAttacker(ABC):
    """Abstract base class for adversarial attackers.
    
    All attackers (e.g., PGD, FGSM, Fog, Rain) must inherit from this class.
    Attackers should rely on the BaseModel interface (specifically get_logits)
    to interact with the model, ensuring decoupling from specific model architectures.
    """
    
    def __init__(self, **kwargs):
        """Initialize the attacker.
        
        Args:
            **kwargs: Configuration parameters for the attacker (e.g., epsilon, steps).
        """
        self.cfg = kwargs

    @abstractmethod
    def generate(self, model: BaseModel, images: torch.Tensor, targets: Any, **kwargs) -> torch.Tensor:
        """Generate adversarial examples.
        
        Args:
            model (BaseModel): The victim model. Must implement get_logits().
            images (torch.Tensor): Original clean images. Shape (B, C, H, W).
                                 Should be normalized as expected by the model.
            targets (Any): Ground truth targets (for untargeted attacks) or 
                           target labels (for targeted attacks).
            **kwargs: Extra arguments specific to the generation process.
            
        Returns:
            torch.Tensor: Adversarial images. Shape (B, C, H, W).
        """
        pass
