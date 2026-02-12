import torch
import torch.optim as optim
from tqdm import tqdm
from typing import Optional, Dict, Union, Tuple, Any
import copy
import os

from core import BaseModel

class Trainer:
    """Generic Trainer Pipeline for BaseModels.
    
    Handles the training loop, data preprocessing (normalization), and optimization.
    """
    
    def __init__(
        self, 
        model: BaseModel, 
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = "cuda"
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model.to(self.device)

    def train_epoch(self, dataloader: Any, epoch: int = 0) -> Dict[str, float]:
        """Run one epoch of training.
        
        Args:
            dataloader: Iterable dataloader. 
                        Can yield (inputs, targets) tuple or dict (Ultralytics format).
            epoch (int): Current epoch number (for logging).
            
        Returns:
            Dict[str, float]: Dictionary of average losses.
        """
        self.model.train()
        # Use enumerate to get batch_idx
        pbar = tqdm(enumerate(dataloader), desc=f"Epoch {epoch}", total=len(dataloader))
        avg_loss = 0.0
        steps = 0
        
        for batch_idx, batch in pbar:
            # Handle Ultralytics batch format (dict)
            if isinstance(batch, dict) and 'img' in batch:
                inputs = batch['img']
                targets = batch
            elif isinstance(batch, (list, tuple)):
                inputs, targets = batch
            else:
                # Fallback: assume batch itself is inputs? Unlikely.
                raise ValueError(f"Unsupported batch format: {type(batch)}")

            # Move inputs to device and Normalize if needed
            inputs = inputs.to(self.device, non_blocking=True)
            
            if inputs.dtype == torch.uint8:
                inputs = inputs.float() / 255.0
            
            if batch_idx == 0:
                 print(f"Inputs: Shape={inputs.shape}, Range=[{inputs.min():.3f}, {inputs.max():.3f}]")

            # Move targets to device if they are tensors inside a dict
            if isinstance(targets, dict):
                for k, v in targets.items():
                    if isinstance(v, torch.Tensor):
                        targets[k] = v.to(self.device, non_blocking=True)
            elif isinstance(targets, torch.Tensor):
                targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            
            # Forward pass
            # Note: inputs are now (B, C, H, W) float32 [0, 1]
            preds = self.model(inputs)
            
            # Compute Loss
            # compute_loss should handle the specific format of targets# Compute Loss
            loss_dict = self.model.compute_loss(preds, targets)
            loss = loss_dict['loss']
            
            # Print detailed loss occasionally
            if batch_idx % 10 == 0:
                # Debug logging
                if 'box_loss' in loss_dict:
                    box = loss_dict['box_loss'].item() if hasattr(loss_dict['box_loss'], 'item') else loss_dict['box_loss']
                    cls = loss_dict['cls_loss'].item() if hasattr(loss_dict['cls_loss'], 'item') else loss_dict['cls_loss']
                    dfl = loss_dict['dfl_loss'].item() if hasattr(loss_dict['dfl_loss'], 'item') else loss_dict['dfl_loss']
                    print(f"Batch {batch_idx}: Loss={loss.item():.4f} (Box={box:.4f}, Cls={cls:.4f}, DFL={dfl:.4f})")
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Logging
            current_loss = loss.item()
            avg_loss += current_loss
            steps += 1
            
            # Update progress bar with main loss
            pbar.set_postfix({"loss": f"{current_loss:.4f}"})
        
        return {"train_loss": avg_loss / steps if steps > 0 else 0.0}

    def validate(self, data_cfg_path: str, batch_size: int = 16, imgsz: int = 640) -> Dict[str, float]:
        """Run validation using Ultralytics Validator.
        
        Args:
            data_cfg_path (str): Path to data.yaml.
            batch_size (int): Batch size.
            imgsz (int): Image size.
            
        Returns:
            Dict[str, float]: Metrics dictionary (mAP50, mAP50-95).
        """
        # Robust Validation Strategy: Save -> Load -> Validate
        # This avoids issues with deepcopying complex YOLO objects or in-place modifications during val()
        
        temp_pt = f"temp_val_{os.getpid()}.pt"
        try:
            # 1. Save current model state
            # We need to save it in a way Ultralytics YOLO can load (full model or state dict with structure)
            # The easiest way is to use the wrapper's yolo object to save, 
            # BUT we must ensure self.model.yolo.model IS our trained self.model.pytorch_model
            
            # Ensure synchronization
            if hasattr(self.model, 'yolo'):
                # Make sure the YOLO wrapper points to the trained PyTorch model
                self.model.yolo.model = self.model.pytorch_model
                
                # Save using YOLO's save method to ensure correct format (pickled model + args)
                # self.model.yolo.save(temp_pt) -> This might save the original file?
                # Let's manually save using torch.save but in Ultralytics compatible format?
                # Or just use the YOLO object.
                
                # Using torch.save on the DetectionModel (self.model.pytorch_model) directly
                # Ultralytics can load a .pt file if it contains a 'model' key or is the model itself.
                torch.save({'model': self.model.pytorch_model, 'train_args': getattr(self.model.yolo, 'overrides', {})}, temp_pt)
                
                # 2. Load into a fresh YOLO instance
                from ultralytics import YOLO
                val_model = YOLO(temp_pt, task='detect')
                
                # 3. Run Validation
                results = val_model.val(
                    data=data_cfg_path,
                    batch=batch_size,
                    imgsz=imgsz,
                    device=self.device,
                    verbose=False,
                    plots=False
                )
                
                return {
                    "mAP50": results.box.map50,
                    "mAP50-95": results.box.map,
                }
                
        except Exception as e:
            print(f"Validation failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
        finally:
            # Cleanup
            if os.path.exists(temp_pt):
                os.remove(temp_pt)
        
        return {}
