import torch
import torch.nn.functional as F
from core.registry import ATTACKERS
from core.abstract_attacker import BaseAttacker
from core.abstract_model import BaseModel
from torchvision.transforms import GaussianBlur

from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import xywh2xyxy, scale_boxes
from ultralytics.utils.metrics import box_iou

@ATTACKERS.register_module("attention_fog")
class AttentionFogAttacker(BaseAttacker):
    """
    Attention-Guided Atmospheric Scattering Fog Attacker.
    Uses model gradient (saliency) as a proxy for 'depth' or 'density' in the physical fog equation.
    Equation: I_adv = I_clean * t + A * (1 - t)
    where t = exp(-beta * density_map)
    density_map is derived from smoothed gradient magnitude.
    """
    def __init__(self, A=1.0, kernel_size=41, sigma=10.0, steps=30, alpha=0.01, max_density=0.9, **kwargs):
        super().__init__(**kwargs)
        self.A = A        # Atmospheric light intensity (0-1)
        self.steps = steps
        self.alpha = alpha # Step size for density accumulation (merged with beta)
        self.max_density = max_density
        # Smoothing parameters to make fog look natural/clumpy rather than noisy
        self.blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def check_success(self, preds, targets, shape):
        """
        Check if all targets are missed (early stopping).
        Returns True if all targets are missed (success).
        """
        return False # Disabled early stopping for now as per user request (focus on density constraint)
        
        # Run NMS
        # preds is (B, 4+nc, N) usually
        try:
            # conf_thres=0.25 is standard for detection
            det = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45, max_det=300)
        except Exception:
            # Fallback if preds structure is different
            return False
            
        # Check matches for each image in batch
        all_missed = True
        
        # targets['bboxes'] is (M, 4) normalized cx,cy,w,h
        # targets['batch_idx'] is (M,)
        gt_bboxes = targets.get('bboxes', [])
        batch_idx = targets.get('batch_idx', [])
        
        if len(gt_bboxes) == 0:
            return True # No targets to detect
            
        B = len(det)
        for i in range(B):
            pred_boxes = det[i][:, :4] # x1, y1, x2, y2
            
            # Get GT for this image
            mask = (batch_idx == i)
            if not mask.any():
                continue
                
            img_gt = gt_bboxes[mask]
            # Convert GT to xyxy and pixel coords
            # img_gt is normalized cx, cy, w, h
            H, W = shape[2], shape[3]
            img_gt_xyxy = xywh2xyxy(img_gt)
            img_gt_xyxy[:, [0, 2]] *= W
            img_gt_xyxy[:, [1, 3]] *= H
            
            if len(pred_boxes) == 0:
                continue # All missed for this image
                
            # Ensure device consistency
            if img_gt_xyxy.device != pred_boxes.device:
                img_gt_xyxy = img_gt_xyxy.to(pred_boxes.device)
                
            # Check IoU
            iou = box_iou(img_gt_xyxy, pred_boxes)
            # If any GT has a match > 0.5, then it's detected
            # iou shape (M, N)
            max_iou, _ = iou.max(1) # Max IoU for each GT
            if (max_iou > 0.5).any():
                all_missed = False
                break
                
        return all_missed

    def get_undetected_mask(self, preds, targets, shape):
        """
        Returns a mask (B, 1, H, W) where 1 means 'allow fog' and 0 means 'no fog'.
        Areas of undetected ground truth objects are set to 0.
        """
        B, C, H, W = shape
        # Default to ones (allow fog everywhere)
        device = preds[0].device if isinstance(preds, (list, tuple)) else preds.device
        mask_map = torch.ones((B, 1, H, W), device=device)
        total_undetected = 0
        
        try:
            det = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45, max_det=300)
        except Exception:
            return mask_map, 0
            
        gt_bboxes = targets.get('bboxes', [])
        batch_idx = targets.get('batch_idx', [])
        
        if len(gt_bboxes) == 0:
            return mask_map, 0
            
        for i in range(B):
            mask = (batch_idx == i)
            if not mask.any():
                continue
                
            img_gt = gt_bboxes[mask]
            # Convert GT to xyxy and pixel coords
            img_gt_xyxy = xywh2xyxy(img_gt)
            img_gt_xyxy[:, [0, 2]] *= W
            img_gt_xyxy[:, [1, 3]] *= H
            
            if img_gt_xyxy.device != device:
                img_gt_xyxy = img_gt_xyxy.to(device)
                
            pred_boxes = det[i][:, :4]
            
            undetected_indices = []
            if len(pred_boxes) == 0:
                undetected_indices = range(len(img_gt_xyxy))
            else:
                iou = box_iou(img_gt_xyxy, pred_boxes)
                max_iou, _ = iou.max(1)
                undetected_indices = (max_iou <= 0.5).nonzero().flatten()
            
            total_undetected += len(undetected_indices)
            
            for idx in undetected_indices:
                box = img_gt_xyxy[idx]
                x1, y1, x2, y2 = box.int()
                x1 = max(0, min(x1.item(), W))
                x2 = max(0, min(x2.item(), W))
                y1 = max(0, min(y1.item(), H))
                y2 = max(0, min(y2.item(), H))
                
                if x2 > x1 and y2 > y1:
                    mask_map[i, 0, y1:y2, x1:x2] = 0.0
                    
        return mask_map, total_undetected

    def generate(self, model, images, targets, **kwargs) -> torch.Tensor:
        """
        Args:
            model: Victim model (must support compute_loss)
            images: (B, C, H, W) normalized [0, 1]
            targets: Batch dict for loss computation
        """
        clean_images = images.clone().detach()
        adv_images = images.clone().detach().requires_grad_(True)
        
        # Initialize density map
        density_map = torch.zeros_like(images[:, 0:1, :, :])
        
        for step in range(self.steps):
            # 1. Forward Pass to get Loss & Preds
            # We need to handle model mode. 
            # Ideally we want grads, so train mode or eval with grad?
            # YOLOv8Wrapper usually handles this.
            
            # Check success (Early Stopping)
            # We use no_grad for inference check to save memory/time
            with torch.no_grad():
                preds_val = model(adv_images)
                if self.check_success(preds_val, targets, images.shape):
                    # print(f"Attack success at step {step}")
                    break
            
            # Enable grad for optimization
            if adv_images.grad is not None:
                adv_images.grad.zero_()
            adv_images.requires_grad_(True)
            
            preds = model(adv_images)
            loss_dict = model.compute_loss(preds, targets)
            loss = loss_dict['loss']
            
            # 2. Compute Gradients (Saliency Map)
            grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
            
            # 3. Process Gradient to Density Map Update
            # Take max absolute gradient across channels: (B, 1, H, W)
            saliency = torch.max(torch.abs(grad), dim=1, keepdim=True)[0]
            
            # Normalize saliency per image
            B = saliency.shape[0]
            for i in range(B):
                s_min = saliency[i].min()
                s_max = saliency[i].max()
                if s_max > s_min:
                    saliency[i] = (saliency[i] - s_min) / (s_max - s_min)
                else:
                    saliency[i] = 0.0
                    
            # Smooth the update
            update = self.blur(saliency)
            
            # Accumulate density
            # We add to the existing density map
            # Use alpha to control step size
            density_map = density_map + self.alpha * update
            
            # Constrain density map to max_density
            density_map = torch.clamp(density_map, 0, self.max_density)
            
            # 4. Apply Atmospheric Scattering Equation
            # Transmission t(x) = exp(-density)
            transmission = torch.exp(-density_map)
            
            # I_fog = I_clean * t + A * (1 - t)
            # Note: We apply to CLEAN images every time to maintain physical consistency
            adv_images = clean_images * transmission + self.A * (1 - transmission)
            adv_images = torch.clamp(adv_images, 0, 1)
            adv_images = adv_images.detach()
        
        if kwargs.get('return_transmission', False):
            # Recalculate transmission for the final density map
            transmission = torch.exp(-density_map)
            return adv_images, transmission

        return adv_images
