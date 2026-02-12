
import os
import argparse
import torch
from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np

from core import MODELS, ATTACKERS
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import xywh2xyxy, scale_boxes
from ultralytics.utils.metrics import ap_per_class, box_iou

# Import modules to register them
import models.yolo_wrapper
import attacks.fgsm
import attacks.pgd
import attacks.physical
import attacks.diffusion
import attacks.attention_fog
import attacks.sgsf

class AttackPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def build_model(self):
        """Build and load the victim model."""
        # Load data config to get class info
        data_cfg = check_det_dataset(self.cfg.data)
        num_classes = data_cfg.get('nc')
        class_names = data_cfg.get('names')
        
        model_cfg = {
            'type': self.cfg.model_type,
            'model_path': self.cfg.model_path,
            'device': self.cfg.device,
            'num_classes': num_classes,
            'names': class_names
        }
        self.model = MODELS.build(model_cfg)
        self.model.eval() # Set to eval mode initially
        print(f"Model {self.cfg.model_type} loaded from {self.cfg.model_path}")

    def build_attacker(self):
        """Build the attacker."""
        attacker_cfg = {
            'type': self.cfg.attacker,
            'epsilon': self.cfg.epsilon,
            'offload_cpu': self.cfg.offload_cpu,
            'beta': self.cfg.beta,
            'A': self.cfg.A,
            'N': self.cfg.N,
            'occlusion_ratio': self.cfg.occlusion_ratio,
        }
        self.attacker = ATTACKERS.build(attacker_cfg)
        print(f"Attacker {self.cfg.attacker} built with epsilon={self.cfg.epsilon}")

    def build_dataloader(self):
        """Build the dataloader."""
        # We use Ultralytics data loading tools
        data_cfg = check_det_dataset(self.cfg.data)
        self.data_cfg = data_cfg
        
        # Build dataset (use validation set for attack generation usually, or train)
        # If mode is 'generate', we might want to generate for the whole dataset
        split = self.cfg.split 
        img_path = data_cfg[split]
        
        # Ensure cfg has required attributes for build_yolo_dataset
        if not hasattr(self.cfg, 'rect'):
            self.cfg.rect = False
        if not hasattr(self.cfg, 'fraction'):
            self.cfg.fraction = 1.0
        if not hasattr(self.cfg, 'single_cls'):
            self.cfg.single_cls = False
        if not hasattr(self.cfg, 'cache'):
            self.cfg.cache = False
        if not hasattr(self.cfg, 'task'):
            self.cfg.task = 'detect'
        if not hasattr(self.cfg, 'classes'):
            self.cfg.classes = None
        if not hasattr(self.cfg, 'mask_ratio'):
            self.cfg.mask_ratio = 4
        if not hasattr(self.cfg, 'overlap_mask'):
            self.cfg.overlap_mask = True
        if not hasattr(self.cfg, 'box'):
            self.cfg.box = True # Needed for box loss? No, for dataset probably.
        if not hasattr(self.cfg, 'project'):
            self.cfg.project = 'runs'
        if not hasattr(self.cfg, 'name'):
            self.cfg.name = 'exp'
        if not hasattr(self.cfg, 'exist_ok'):
            self.cfg.exist_ok = False
            
        # We reuse build_dataloader from train.py logic or simplified
        # For simplicity, we can use build_yolo_dataset and build_dataloader
        
        # Note: build_yolo_dataset returns a YOLODataset
        self.dataset = build_yolo_dataset(
            self.cfg, 
            img_path, 
            batch=self.cfg.batch_size, 
            data=data_cfg, 
            mode='val', # Use val mode to avoid mosaic augmentation during generation? 
                        # Actually for FGSM we want clean images. 'val' mode disables most augs.
            rect=False, 
            stride=32
        )
        
        self.loader = build_dataloader(
            self.dataset, 
            batch=self.cfg.batch_size, 
            workers=self.cfg.workers, 
            shuffle=False, 
            rank=-1
        )
        print(f"Dataloader built for {split} split with {len(self.dataset)} images.")

    def compute_and_print_metrics(self, stats, prefix=""):
        if len(stats) == 0:
            print(f"{prefix} No stats to compute.")
            return

        try:
            stats_concat = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
            tp, conf, pred_cls, target_cls = stats_concat
            
            # Ensure tp is float
            tp = torch.tensor(tp).float()
            conf = torch.tensor(conf)
            pred_cls = torch.tensor(pred_cls)
            target_cls = torch.tensor(target_cls)

            results = ap_per_class(tp, conf, pred_cls, target_cls, plot=False)
            ap = results[5]
            
            # Debug shape
            # print(f"DEBUG: {prefix} AP shape: {ap.shape}")
            
            if ap.ndim == 1:
                ap = ap.reshape(1, -1)
            
            map50 = ap[:, 0].mean()
            map50_95 = ap.mean()
            
            print(f"{prefix} mAP50: {map50:.4f}")
            print(f"{prefix} mAP50-95: {map50_95:.4f}")
            
        except Exception as e:
            print(f"{prefix} Error computing metrics: {e}")

    def run(self):
        """Run the attack generation pipeline."""
        self.build_model()
        self.build_attacker()
        self.build_dataloader()
        
        print("Starting adversarial example generation...")
        
        # Stats for mAP calculation
        stats = [] # (correct, conf, pred_cls, target_cls)
        clean_stats = []
        
        count = 0
        pbar = tqdm(self.loader)
        for batch_i, batch in enumerate(pbar):
            # Batch is a dict with 'img', 'cls', 'bboxes', 'batch_idx', 'im_file', etc.
            images = batch['img'].to(self.device).float() / 255.0
            
            # Prepare targets for loss computation
            # Ultralytics loss expects batch dict
            batch['img'] = images # Update with normalized images on device
            
            # 1. Evaluate Clean Images
            self.update_metrics(images, batch, clean_stats)

            # Generate adversarial images
            # Pass steps and alpha if available in cfg
            kwargs = {}
            if hasattr(self.cfg, 'steps'):
                kwargs['steps'] = self.cfg.steps
            if hasattr(self.cfg, 'alpha'):
                kwargs['alpha'] = self.cfg.alpha
            
            # If we want to save at original resolution, we need the transmission map
            if self.cfg.save_resolution == 'original':
                kwargs['return_transmission'] = True
                
            adv_result = self.attacker.generate(self.model, images, batch, **kwargs)
            
            if self.cfg.save_resolution == 'original' and isinstance(adv_result, tuple):
                adv_images, transmission = adv_result
            else:
                adv_images = adv_result
                transmission = None
            
            # Evaluate current batch
            self.update_metrics(adv_images, batch, stats)
            
            # Calculate current mAP (Adversarial)
            try:
                if len(stats) > 0:
                    stats_concat = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
                    tp, conf, pred_cls, target_cls = stats_concat
                    results = ap_per_class(torch.tensor(tp).float(), torch.tensor(conf), torch.tensor(pred_cls), torch.tensor(target_cls), plot=False)
                    ap = results[5]
                    if ap.ndim == 1:
                        ap = ap.reshape(1, -1)
                    map50 = ap[:, 0].mean()
                    map50_95 = ap.mean()
                    pbar.set_description(f"Adv mAP50: {map50:.4f}")
            except Exception:
                pass
            
            # Save images
            self.save_batch(adv_images, batch['im_file'], batch['ori_shape'], transmission=transmission)
            
            count += len(images)
            if self.cfg.limit > 0 and count >= self.cfg.limit:
                print(f"Reached limit of {self.cfg.limit} images.")
                break
        
        # Final Report
        print("\nFinal Metrics:")
        self.compute_and_print_metrics(clean_stats, prefix="[Clean]")
        self.compute_and_print_metrics(stats, prefix="[Adversarial]")

    def update_metrics(self, images, batch, stats):
        """Run inference and update stats for mAP calculation."""
        self.model.eval()
        with torch.no_grad():
            preds = self.model(images)
            
            # NMS
            # preds is usually (B, 4+nc, N) or tuple
            # If tuple, it's (preds, features) -> use preds[0]
            if isinstance(preds, (tuple, list)):
                preds = preds[0]
                
            det = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45, max_det=300)
            
            # Process each image
            gt_bboxes = batch.get('bboxes', []) # Normalized cx,cy,w,h
            gt_cls = batch.get('cls', [])
            batch_idx = batch.get('batch_idx', [])
            
            for i, pred in enumerate(det):
                # pred: (N, 6) -> x1,y1,x2,y2,conf,cls
                
                # Get GT
                mask = (batch_idx == i)
                if not mask.any():
                    # No GT for this image
                    # If detections exist, they are all FP
                    if len(pred) > 0:
                        # Append stats for FP?
                        # ap_per_class expects matches.
                        # We need to handle images with no GT but with detections.
                        # target_cls for this image is empty.
                        pass
                    continue
                    
                img_gt_bbox = gt_bboxes[mask]
                img_gt_cls = gt_cls[mask]
                
                # Convert GT to pixels
                H, W = images.shape[2], images.shape[3]
                img_gt_bbox_xyxy = xywh2xyxy(img_gt_bbox)
                img_gt_bbox_xyxy[:, [0, 2]] *= W
                img_gt_bbox_xyxy[:, [1, 3]] *= H
                
                # Ensure device match
                if img_gt_bbox_xyxy.device != pred.device:
                    img_gt_bbox_xyxy = img_gt_bbox_xyxy.to(pred.device)
                if img_gt_cls.device != pred.device:
                    img_gt_cls = img_gt_cls.to(pred.device)

                # Match predictions
                correct = self.process_batch(pred, img_gt_bbox_xyxy, img_gt_cls)
                
                # correct shape: (n_pred, 10) bool
                if len(pred) > 0:
                    stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), img_gt_cls.cpu()))
                else:
                    # No predictions, but there are GTs.
                    # This counts as FN.
                    # stats needs to account for missed GTs?
                    # ap_per_class uses target_cls to count total GTs.
                    # But here we append (correct, conf, pred_cls, target_cls).
                    # If no preds, we can't append correct/conf/pred_cls.
                    # But we MUST append target_cls somewhere?
                    # Actually, ap_per_class logic:
                    # It iterates over unique classes.
                    # n_gt = (target_cls == c).sum()
                    # So we need to accumulate ALL target_cls.
                    # But the signature is (tp, conf, pred_cls, target_cls).
                    # These arrays must be aligned?
                    # No, target_cls is the list of ALL targets seen so far?
                    # NO. 
                    # Let's check ap_per_class doc/implementation.
                    # "target_cls: (n_targets, )"
                    # "pred_cls: (n_preds, )"
                    # They are NOT aligned.
                    # So we just append the GT classes for this image to the list.
                    # But my stats structure: `stats.append((correct, conf, pred_cls, img_gt_cls))`
                    # This implies I'm appending a tuple per image.
                    # Then I concat them.
                    # So target_cls will be concatenation of all img_gt_cls.
                    # pred_cls will be concatenation of all pred_cls.
                    # This is correct.
                    pass
                    
                    # If len(pred) == 0, we still need to record that we saw these GTs.
                    # But stats structure is designed for when we have predictions.
                    # If no predictions, tp/conf/pred_cls are empty.
                    # But target_cls is not.
                    
                    if len(pred) == 0:
                         stats.append((torch.zeros(0, 10, dtype=torch.bool), torch.zeros(0), torch.zeros(0), img_gt_cls.cpu()))
    
    def process_batch(self, detections, labels, labels_cls):
        """
        Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
        Arguments:
            detections (Tensor): (N, 6), x1, y1, x2, y2, conf, cls
            labels (Tensor): (M, 4), x1, y1, x2, y2
            labels_cls (Tensor): (M, )
        Returns:
            correct (Tensor): (N, 10), for 10 IoU levels .50:.05:.95
        """
        iou = box_iou(labels, detections[:, :4])
        correct = torch.zeros(detections.shape[0], 10, dtype=torch.bool, device=iou.device)
        labels_cls = labels_cls.view(-1)
        correct_class = labels_cls[:, None] == detections[:, 5]
        
        # IoU thresholds
        iouv = torch.linspace(0.5, 0.95, 10, device=iou.device)
        
        for i, threshold in enumerate(iouv):
            # iou >= threshold and correct class
            matches = (iou >= threshold) & correct_class
            # For each detection, find if it matches any GT
            # We need to solve bipartite matching or greedy
            # Ultralytics uses greedy matching (max IoU)
            
            # For each detection column, find max IoU with correct class
            if matches.shape[0] > 0:
                # matches is (M_gt, N_pred)
                # We want to find matches. 
                # Ideally: detection j matches GT i.
                # A detection can match at most one GT.
                # A GT can be matched by at most one detection.
                
                # Simplified greedy:
                # 1. Find all matches > threshold
                # 2. Sort by IoU desc
                # 3. Assign
                
                # Ultralytics internal implementation does this.
                # I'll use a simpler approximation if strict matching is hard to code quickly.
                # Actually, let's just check if *any* GT matches.
                # But multiple detections on same GT is FP.
                
                # Let's try to do it right.
                iou_vals = iou * correct_class # Zero out wrong classes
                
                if iou_vals.numel() > 0:
                    matches_idx = torch.nonzero(iou_vals >= threshold) # (n_matches, 2) -> (gt_idx, pred_idx)
                    if matches_idx.shape[0] > 0:
                        # Get IoU values for these matches
                        match_ious = iou_vals[matches_idx[:, 0], matches_idx[:, 1]]
                        # Sort by IoU desc
                        matches_idx = matches_idx[torch.argsort(match_ious, descending=True)]
                        
                        seen_gt = set()
                        seen_pred = set()
                        
                        matches_idx_np = matches_idx.cpu().numpy()
                        if len(matches_idx_np.shape) == 2:
                            for row in matches_idx_np:
                                gt_idx = int(row[0])
                                pred_idx = int(row[1])
                                if gt_idx not in seen_gt and pred_idx not in seen_pred:
                                    correct[pred_idx, i] = True
                                    seen_gt.add(gt_idx)
                                    seen_pred.add(pred_idx)
                        else:
                             print(f"Unexpected matches_idx shape: {matches_idx_np.shape}")
                                
        return correct

    def save_batch(self, adv_images, im_files, ori_shapes, transmission=None):
        """Save adversarial images to disk."""
        # adv_images: (B, C, H, W) normalized [0, 1]
        
        if transmission is None or self.cfg.save_resolution != 'original':
             # Save the resized (640x640) images directly
            adv_images = adv_images.detach().cpu().numpy()
            # Convert to [0, 255] and HWC
            adv_images = (adv_images * 255).astype(np.uint8)
            
            for i, img_file in enumerate(im_files):
                filename = os.path.basename(img_file)
                parent = os.path.basename(os.path.dirname(img_file))
                
                save_path = self.save_dir / parent / filename
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                img = adv_images[i].transpose(1, 2, 0) # CHW -> HWC
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(save_path), img)
        else:
            # High-Resolution Saving Mode
            # 1. Load original image
            # 2. Resize transmission map to original size
            # 3. Apply fog equation
            
            for i, img_file in enumerate(im_files):
                filename = os.path.basename(img_file)
                parent = os.path.basename(os.path.dirname(img_file))
                
                save_path = self.save_dir / parent / filename
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Load Original Image (H_orig, W_orig, 3) BGR
                orig_img = cv2.imread(img_file)
                if orig_img is None:
                    print(f"Warning: Could not read original image {img_file}")
                    continue
                    
                H_orig, W_orig = orig_img.shape[:2]
                
                # Get Transmission Map for this image (1, H, W) -> (H, W)
                t_map = transmission[i].detach().cpu().numpy().squeeze()
                
                # Resize Transmission Map to (W_orig, H_orig)
                # cv2.resize expects (W, H)
                t_map_high = cv2.resize(t_map, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
                
                # Expand dims for broadcasting (H, W, 1)
                t_map_high = t_map_high[:, :, np.newaxis]
                
                # Normalize Original Image to [0, 1]
                orig_img_norm = orig_img.astype(np.float32) / 255.0
                
                # Apply Fog Equation
                # I_adv = I * t + A * (1 - t)
                # Note: orig_img is BGR, A should be applied carefully if it's not white
                # Assuming A is white (1.0) or gray. If A is scalar, it works for all channels.
                A = self.cfg.A
                
                adv_img_high = orig_img_norm * t_map_high + A * (1 - t_map_high)
                
                # Clip and Convert back to uint8
                adv_img_high = np.clip(adv_img_high * 255.0, 0, 255).astype(np.uint8)
                
                cv2.imwrite(str(save_path), adv_img_high)

def parse_args():
    parser = argparse.ArgumentParser(description='Adversarial Attack Generation Pipeline')
    parser.add_argument('--model-type', type=str, default='yolov8', help='Model type registered in core')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, default='cityscapes.yaml', help='Data yaml file')
    parser.add_argument('--attacker', type=str, default='fgsm', help='Attacker name registered in core')
    parser.add_argument('--epsilon', type=float, default=0.01, help='Perturbation magnitude')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--save-dir', type=str, default='runs/attack/exp', help='Directory to save adversarial images')
    parser.add_argument('--split', type=str, default='train', help='Data split to generate (train/val/test)')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of images to generate (0 for all)')
    parser.add_argument('--offload-cpu', action='store_true', help='Offload diffusion model to CPU to save VRAM')
    parser.add_argument('--beta', type=float, default=0.9, help='Beta for fog/scattering')
    parser.add_argument('--A', type=float, default=1.0, help='Atmospheric light')
    parser.add_argument('--N', type=int, default=20, help='Number of candidates for SGSF')
    parser.add_argument('--occlusion-ratio', type=float, default=0.1, help='Occlusion ratio for SGSF')
    parser.add_argument('--steps', type=int, default=30, help='Number of steps for iterative attacks')
    parser.add_argument('--alpha', type=float, default=1.0, help='Step size for iterative attacks')
    parser.add_argument('--max-density', type=float, default=0.9, help='Max density for attention fog')
    parser.add_argument('--save-resolution', type=str, default='model', choices=['model', 'original'], 
                        help='Save images at model resolution (640x640) or original resolution (upsample fog)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    pipeline = AttackPipeline(args)
    pipeline.run()
