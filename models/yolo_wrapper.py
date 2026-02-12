from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss

from core import MODELS, BaseModel

@MODELS.register_module("yolov8")
class Yolov8Wrapper(BaseModel):
    """Transparent Proxy for YOLOv8 models using Ultralytics."""
    
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu", num_classes: int = None, names: Dict[int, str] = None, **kwargs):
        super().__init__()
        self.model_path = model_path
        self.device_str = device
        
        # Load the model using Ultralytics YOLO wrapper
        self.yolo = YOLO(model_path)
        
        # Check if we need to update the model head for different number of classes
        if num_classes is not None and hasattr(self.yolo.model, 'nc') and self.yolo.model.nc != num_classes:
            print(f"Transfer Learning: Updating model head from {self.yolo.model.nc} to {num_classes} classes...")
            
            # Load weights from the current model
            weights = self.yolo.model.state_dict()
            
            # Determine config name based on model path (e.g., yolov8m.pt -> yolov8m.yaml)
            import os
            cfg_name = os.path.basename(model_path).replace('.pt', '.yaml')
            
            # Create a new DetectionModel with the correct number of classes
            try:
                from ultralytics.nn.tasks import DetectionModel
                from ultralytics.utils.torch_utils import intersect_dicts
                
                # We need to make sure the cfg file is available. 
                # Ultralytics can load standard configs by name.
                new_model = DetectionModel(cfg=cfg_name, nc=num_classes)
                
                # Transfer weights
                csd = intersect_dicts(weights, new_model.state_dict())
                print(f"Transferred {len(csd)}/{len(new_model.state_dict())} items from pretrained model.")
                
                # Debug: Check what's missing
                model_keys = set(new_model.state_dict().keys())
                csd_keys = set(csd.keys())
                missing = model_keys - csd_keys
                if missing:
                    print(f"Missing keys ({len(missing)}): {list(missing)[:5]} ...")
                
                new_model.load_state_dict(csd, strict=False)
                
                # Attach args from the original model to the new model
                # v8DetectionLoss requires model.args for hyperparameters
                if hasattr(self.yolo.model, 'args'):
                    new_model.args = self.yolo.model.args
                else:
                    # Fallback to default args if original model doesn't have them
                    # These are standard YOLOv8 hyps
                    new_model.args = {
                        'box': 7.5, 'cls': 0.5, 'dfl': 1.5,
                        'pose': 12.0, 'kobj': 1.0, 'label_smoothing': 0.0,
                        'nbs': 64, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
                        'degrees': 0.0, 'translate': 0.1, 'scale': 0.5,
                        'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0,
                        'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.0,
                        'copy_paste': 0.0
                    }
                
                # Replace the model
                self.yolo.model = new_model
                print("Model head updated successfully.")
                
            except Exception as e:
                print(f"Error updating model head: {e}")
                print("Trying fallback: rely on Ultralytics to handle it via overrides if possible (unlikely for pure model usage).")
        
        # Extract the underlying PyTorch model (DetectionModel)
        self.pytorch_model = self.yolo.model
        self.pytorch_model.to(self.device_str)
        
        # Set class names if provided
        if names:
            self.pytorch_model.names = names
            self.yolo.model.names = names

        
        # Initialize loss function
        # v8DetectionLoss requires the model to be on the correct device during init
        # to set up anchors/device correctly.
        self.criterion = v8DetectionLoss(self.pytorch_model)
        # Manually attach hyperparameters if missing (sometimes happens with raw model extraction)
        if not hasattr(self.criterion, 'hyp'):
            # Fallback to default/args
             self.criterion.hyp = self.yolo.overrides if hasattr(self.yolo, 'overrides') else {}
        
        # Ensure 'box', 'cls', 'dfl' exist in hyp
        if isinstance(self.criterion.hyp, dict):
             # Ultralytics 8.4.0+ uses a BoxNamespace object or dict. 
             # Let's check if we need to wrap it or access it differently.
             # The error is: 'dict' object has no attribute 'box'
             # This means self.criterion.hyp is a dict, but the code expects an object with attributes.
             # Solution: Convert dict to a simple namespace/object
             class DictToObj:
                 def __init__(self, d):
                     self.__dict__ = d
                     # Set defaults if missing
                     if 'box' not in d: self.box = 7.5
                     if 'cls' not in d: self.cls = 0.5
                     if 'dfl' not in d: self.dfl = 1.5
             
             self.criterion.hyp = DictToObj(self.criterion.hyp)

    def train(self, mode: bool = True):
        """Set the model to training mode.
        
        Note: We MUST avoid calling self.yolo.train() because that triggers the 
        full Ultralytics training pipeline. We only want to set the underlying 
        PyTorch modules to training mode.
        """
        # Skip calling super().train(mode) if it delegates to something that breaks
        # But BaseModel inherits from nn.Module, so super().train() is nn.Module.train()
        # The issue is that self.yolo is a YOLO object which has a train() method 
        # that is NOT nn.Module.train(), but a high-level task launcher.
        # However, Yolov8Wrapper inherits from BaseModel -> nn.Module.
        # So super().train() should be fine unless BaseModel does something weird.
        # Wait, if self.yolo is assigned to self, does it shadow? No.
        
        # The traceback shows:
        # File "/root/autodl-tmp/dky/foggy_test/models/yolo_wrapper.py", line 33, in train
        #    super().train(mode)
        # File .../torch/nn/modules/module.py", line 2891, in train
        #    module.train(mode)
        # File .../ultralytics/engine/model.py", line 769, in train
        
        # This means one of the submodules is the YOLO object itself?
        # In __init__: self.yolo = YOLO(model_path)
        # self.yolo IS NOT an nn.Module. It's a Python class wrapper.
        # But if we assigned it as an attribute of an nn.Module (Yolov8Wrapper), 
        # PyTorch doesn't register it as a submodule unless it inherits from nn.Module.
        # Ultralytics YOLO class does NOT inherit from nn.Module.
        # Wait, let's check why module.train(mode) calls ultralytics/engine/model.py train.
        
        # Ah, if self.yolo is NOT an nn.Module, it shouldn't be in self.modules() loop 
        # inside nn.Module.train().
        # UNLESS Ultralytics YOLO class actually DOES inherit from nn.Module?
        # Let's check imports. from ultralytics import YOLO.
        # Usually YOLO("...") returns a Model object which inherits from simpleBaseModel?
        # No, YOLO class in ultralytics/models/yolo/model.py inherits from Model.
        # ultralytics.engine.model.Model inherits from object? Or nn.Module?
        
        # If Ultralytics Model inherits from nn.Module, then self.yolo is a submodule.
        # And when we call wrapper.train(), it calls self.yolo.train().
        # But self.yolo.train() is the high-level trainer launcher, NOT the mode switcher.
        
        # Solution: DO NOT register self.yolo as a submodule.
        # We can store it in a list or just rely on self.pytorch_model which IS the nn.Module.
        # Or just rename self.yolo to something that won't be picked up? 
        # No, standard attributes are not picked up as submodules unless they are nn.Module.
        # So Ultralytics Model MUST be an nn.Module.
        
        # Workaround: Prevent recursion into self.yolo during train().
        # But we can't easily control nn.Module.train() recursion.
        # We should probably NOT store self.yolo as a class attribute if we can avoid it,
        # OR we ensure it's not treated as a submodule.
        # But we need it to extract the model.
        
        # Better: del self.yolo after extraction?
        # Or store it in a way that nn.Module doesn't see?
        
        pass 
        # We will fix this in __init__ by removing self.yolo attribute or making it non-persistent.
        # But for now, let's override train() to manually handle children.
        
        self.training = mode
        # Manually set training mode for the underlying PyTorch model
        self.pytorch_model.train(mode)
        
        # Ensure gradients are enabled for the model parameters
        if mode:
            for p in self.pytorch_model.parameters():
                p.requires_grad = True
                
        # We do NOT call super().train(mode) to avoid calling self.yolo.train()
        # But we should handle other submodules if any. 
        # Since we only have pytorch_model and criterion, let's handle them.
        if hasattr(self, 'criterion') and isinstance(self.criterion, nn.Module):
            self.criterion.train(mode)
        return self

    def eval(self):
        """Set the model to evaluation mode."""
        return self.train(False)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Any:
        """Standard forward pass.
        
        Args:
            x (torch.Tensor): Input images (B, C, H, W). Should be normalized [0, 1].
            
        Returns:
            Tuple[torch.Tensor, Dict]: (concatenated_output, dict_output)
        """
        # Ensure input is on the correct device
        if x.device != next(self.pytorch_model.parameters()).device:
            x = x.to(next(self.pytorch_model.parameters()).device)
            
        return self.pytorch_model(x, *args, **kwargs)

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw logits from the model without NMS.
        
        Args:
            x (torch.Tensor): Input tensor (B, C, H, W).
            
        Returns:
            torch.Tensor: Raw concatenated output (B, 4+nc, anchors).
        """
        preds = self.forward(x)
        if isinstance(preds, (tuple, list)):
            return preds[0]
        return preds

    def compute_loss(self, preds: Any, targets: Any) -> Dict[str, torch.Tensor]:
        """Compute the loss using Ultralytics v8DetectionLoss.
        
        Args:
            preds (Any): Predictions from forward() (Tuple).
            targets (Any): Batch dict containing 'batch_idx', 'cls', 'bboxes'.
                           Format matches Ultralytics dataloader output.

        Returns:
            Dict[str, torch.Tensor]: Loss dictionary.
        """
        # v8DetectionLoss expects (preds, batch)
        # The loss returned by criterion is (total_loss * batch_size, loss_items)
        # We need to mean reduction usually for backward if optimizer expects mean,
        # but Ultralytics loss usually returns sum * batch_size?
        # Let's check v8DetectionLoss.loss implementation:
        # return loss * batch_size, loss_detach
        
        # So 'loss' here is a scalar (tensor) but it might have been multiplied by batch size.
        # If it's a scalar tensor, .backward() works.
        # If it's not a scalar, we get "grad can be implicitly created only for scalar outputs".
        
        loss, loss_items = self.criterion(preds, targets)
        
        # Ensure loss is a scalar (mean reduction if needed, but Ultralytics usually sums)
        if loss.numel() > 1:
            loss = loss.mean()
        
        return {
            "loss": loss,
            "box_loss": loss_items[0],
            "cls_loss": loss_items[1],
            "dfl_loss": loss_items[2]
        }
