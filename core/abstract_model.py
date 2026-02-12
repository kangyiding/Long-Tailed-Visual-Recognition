from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    """Abstract base class for all models in the platform.

    All models (e.g., YOLOv8, ResNet) must inherit from this class to ensure
    compatibility with the training and attack pipelines.
    
    This class enforces a unified interface for:
    1. Getting raw logits (crucial for white-box attacks).
    2. Computing loss (unified interface for training).
    3. Handling different modes (train/val/attack).
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_logits(self, x: torch.Tensor) -> Any:
        """Get raw logits from the model without NMS or Softmax.

        This method is essential for white-box attacks (like PGD, FGSM) where
        gradients need to be computed with respect to the input image.
        The implementation should ensure that the computation graph is maintained
        from input to output.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
                              Values should be normalized as expected by the model.

        Returns:
            Any: Raw output tensor(s). The exact structure depends on the model architecture,
                 but it should be the raw predictions before any post-processing
                 that breaks gradients (like NMS).
        """
        pass

    @abstractmethod
    def compute_loss(self, preds: Any, targets: Any) -> Dict[str, torch.Tensor]:
        """Compute the loss given predictions and targets.

        Args:
            preds (Any): Predictions from the model (output of forward() or get_logits()).
            targets (Any): Ground truth labels. Format depends on the specific model/dataset.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing loss components.
                                     Must contain a key "loss" which is the total loss scalar
                                     used for backpropagation.
        """
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> Any:
        """Standard forward pass.
        
        Depending on the mode (training vs inference), this might return
        different outputs (e.g., loss dict during training, detections during eval).
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            Any: Model output.
        """
        pass
