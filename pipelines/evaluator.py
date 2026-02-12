
import torch
import argparse
from typing import Dict, Any, Optional
from pathlib import Path
import os

from core import MODELS
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg

class Evaluator:
    """Generic Evaluator Pipeline."""
    
    def __init__(self, model_path: str, device: str = "cuda", task: str = "detect"):
        self.model_path = model_path
        self.device = device
        self.task = task
        self.model = None

    def load_model(self):
        """Load the model for evaluation.
        
        For Ultralytics models, we can use the YOLO class directly for validation
        as it provides a robust validator.
        """
        from ultralytics import YOLO
        self.model = YOLO(self.model_path, task=self.task)
        
    def evaluate(self, data_cfg: str, split: str = 'val', batch_size: int = 16, imgsz: int = 640) -> Dict[str, float]:
        """Run evaluation.
        
        Args:
            data_cfg (str): Path to data.yaml.
            split (str): Split to evaluate (val, test).
            batch_size (int): Batch size.
            imgsz (int): Image size.
            
        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        if self.model is None:
            self.load_model()
            
        print(f"Evaluating on {split} set from {data_cfg}...")
        
        # Run validation
        metrics = self.model.val(
            data=data_cfg,
            split=split,
            batch=batch_size,
            imgsz=imgsz,
            device=self.device,
            verbose=True,
            plots=True,
            save=True,
            project='runs/val',
            name='exp'
        )
        
        # Parse metrics
        results = {
            "mAP50": metrics.box.map50,
            "mAP50-95": metrics.box.map,
            "precision": metrics.box.mp,
            "recall": metrics.box.mr,
        }
        
        print(f"Evaluation Results: {results}")
        return results

def parse_args():
    parser = argparse.ArgumentParser(description='Model Evaluation Pipeline')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint (.pt)')
    parser.add_argument('--data', type=str, default='cityscapes.yaml', help='Path to data.yaml')
    parser.add_argument('--split', type=str, default='val', help='Data split to evaluate')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    evaluator = Evaluator(args.model_path, args.device)
    evaluator.evaluate(args.data, args.split, args.batch_size, args.imgsz)
