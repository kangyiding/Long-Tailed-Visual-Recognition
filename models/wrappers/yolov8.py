"""
YOLOv8 Wrapper for Object Detection.
"""
import torch
import torch.nn as nn
from typing import List, Dict, Any, Union
from core.base_model import BaseModel
from core.registry import MODELS
from core.utils.preprocess import DifferentiableLetterbox
try:
    from ultralytics import YOLO
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.ops import non_max_suppression
    from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
except ImportError:
    # Fallback for mocking/testing if ultralytics not installed
    YOLO = None
    v8DetectionLoss = None

@MODELS.register_module("yolov8")
class YOLOv8Wrapper(BaseModel):
    """
    YOLOv8 Wrapper with differentiable preprocessing.
    """
    
    def __init__(self, model_path: str, input_size: tuple = (640, 640), **kwargs):
        """
        Args:
            model_path (str): Path to .pt model.
            input_size (tuple): Target input size (H, W).
            **kwargs: Other config args.
        """
        super().__init__(**kwargs)
        if YOLO is None:
            raise ImportError("Ultralytics not installed.")
            
        self.input_size = input_size
        
        # Load Model
        self.yolo = YOLO(model_path)
        self.model = self.yolo.model
        
        # Differentiable Preprocessing
        # Ultralytics uses 114/255 for pad color (gray)
        self.letterbox = DifferentiableLetterbox(input_size, pad_value=114.0/255.0)
        
        # Loss setup
        # We instantiate v8DetectionLoss. 
        # Note: v8DetectionLoss needs the Hyperparameters (hyp) from the model if available.
        # usually stored in model.args
        self.hyp = self.yolo.overrides if hasattr(self.yolo, 'overrides') else {}
        if not self.hyp and hasattr(self.model, 'args'):
             self.hyp = self.model.args
        
        # Ensure generic hyp exists if empty (required for loss init)
        if not self.hyp:
            # minimal defaults
            self.hyp = {'box': 7.5, 'cls': 0.5, 'dfl': 1.5} 

        self.criterion = v8DetectionLoss(self.model)

    def forward(self, x: torch.Tensor) -> Any:
        """
        Args:
            x (torch.Tensor): RGB input [B, 3, H, W], [0, 1].
        """
        # 1. Differentiable Resize & Pad
        x = self.letterbox(x)
        
        # 2. Forward pass
        # The output format depends on self.model.training
        return self.model(x)

    def get_loss(self, x: torch.Tensor, targets: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute loss for attacks.
        
        Args:
            x (torch.Tensor): Input images.
            targets (List[torch.Tensor]): List of [N, 5] tensors (cls, x, y, w, h).
        """
        # Switch to training mode to get raw head outputs, 
        # but keep BN in eval mode to avoid stat updates.
        training_state = self.model.training
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
                
        try:
            preds = self.forward(x)
            
            # Prepare batch for v8DetectionLoss
            # batch: {'batch_idx':, 'cls':, 'bboxes':, 'device':}
            batch_idx = []
            cls_list = []
            bboxes_list = []
            
            for i, t in enumerate(targets):
                if t is None or t.numel() == 0:
                    continue
                n = t.shape[0]
                batch_idx.append(torch.full((n,), i, device=x.device))
                cls_list.append(t[:, 0:1]) # keep dim
                bboxes_list.append(t[:, 1:])
            
            if not batch_idx:
                return torch.tensor(0.0, device=x.device, requires_grad=True)
                
            batch = {
                'batch_idx': torch.cat(batch_idx),
                'cls': torch.cat(cls_list).view(-1, 1),
                'bboxes': torch.cat(bboxes_list),
                'device': x.device
            }
            
            loss, loss_items = self.criterion(preds, batch)
            return loss
            
        finally:
            # Restore state
            self.model.train(training_state)

    def predict(self, x: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Inference with NMS.
        """
        self.model.eval()
        with torch.no_grad():
            # forward (eval mode -> concatenated output)
            preds = self.forward(x)
            
            # NMS
            # preds is (B, 4+C, N) or similar depending on export
            # Ultralytics non_max_suppression handles it
            
            # Note: The output of model(x) in eval mode might be list or tensor
            # Ultralytics v8 model returns tensor (B, 84, 8400) usually
            
            results = non_max_suppression(preds, 
                                          conf_thres=0.25, 
                                          iou_thres=0.45, 
                                          classes=None)
            
            # Format results
            formatted = []
            for res in results:
                # res: (N, 6) -> x1, y1, x2, y2, conf, cls
                formatted.append({
                    'boxes': res[:, :4],
                    'scores': res[:, 4],
                    'labels': res[:, 5]
                })
            return formatted
