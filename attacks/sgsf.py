import torch
import torch.nn.functional as F
import math
from core.registry import ATTACKERS
from core.abstract_attacker import BaseAttacker
from torchvision.transforms import GaussianBlur
from ultralytics.utils.ops import xywh2xyxy

@ATTACKERS.register_module("sgsf")
class SGSFAttacker(BaseAttacker):
    """
    Semantic-Guided Semantic Fog (SGSF) Attacker.
    
    Key Steps:
    1. Saliency Extraction: Gradient-based heatmap.
    2. Continuous Shape Sampling: Generate natural cloud-like masks using 
       Gaussian noise -> Blur -> Threshold -> Morphology.
    3. Best-of-N Selection: Select the random shape that covers the most saliency 
       within each target's context.
    4. Global Max-Composition: Merge local masks using max operation.
    5. Physical Rendering: Edge feathering and Atmospheric Scattering Equation.
    """
    def __init__(self, beta=3.0, A=1.0, N=10, occlusion_ratio=0.1, context_pad=10, 
                 feather_sigma=5.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.A = A
        self.N = N
        self.occlusion_ratio = occlusion_ratio
        self.context_pad = context_pad
        self.feather_sigma = feather_sigma
        
        # Blur for generating cloud shapes (strong blur for continuity)
        # Kernel size needs to be large enough
        self.cloud_blur = GaussianBlur(kernel_size=21, sigma=5.0)
        
        # Blur for edge feathering (final rendering)
        self.feather_blur = GaussianBlur(kernel_size=21, sigma=feather_sigma)

    def _morphological_closing(self, mask, kernel_size=5):
        """
        Approximate morphological closing using MaxPool (Dilation) then MinPool (Erosion).
        Input: (B, 1, H, W) binary float tensor
        """
        # Dilation = MaxPool
        padding = kernel_size // 2
        dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
        
        # Erosion = -MaxPool(-x)
        # Note: min_pool2d doesn't exist directly with same convenience, implementing via max_pool
        eroded = -F.max_pool2d(-dilated, kernel_size=kernel_size, stride=1, padding=padding)
        
        return eroded

    def _generate_continuous_shape(self, h, w, device):
        """
        Generates N cloud-like shapes of size (h, w).
        Returns: (N, 1, h, w) binary masks
        """
        if h <= 0 or w <= 0:
            return torch.zeros((self.N, 1, h, w), device=device)

        # 1. Gaussian Noise
        noise = torch.randn((self.N, 1, h, w), device=device)
        
        # 2. Strong Blur
        # Need to handle small dimensions where kernel size might be too big
        k_size = 21
        if min(h, w) < k_size:
            k_size = min(h, w)
            if k_size % 2 == 0: k_size -= 1
        
        if k_size >= 3:
            blur_transform = GaussianBlur(kernel_size=k_size, sigma=5.0)
            blurred = blur_transform(noise)
        else:
            blurred = noise

        # 3. Threshold Truncation for Occlusion Ratio
        # Flatten to find quantile
        flat = blurred.view(self.N, -1)
        # We want top 'occlusion_ratio' pixels to be 1
        # quantile value such that P(X > q) = ratio
        # k-th value should be (1 - ratio) * total_pixels
        k = int((1 - self.occlusion_ratio) * flat.shape[1])
        k = max(0, min(k, flat.shape[1] - 1))
        
        thresholds = torch.kthvalue(flat, k, dim=1)[0].view(self.N, 1, 1, 1)
        binary_mask = (blurred > thresholds).float()
        
        # 4. Morphological Closing
        # Apply closing to fill holes and smooth edges
        closed_mask = self._morphological_closing(binary_mask, kernel_size=5)
        
        return closed_mask

    def generate(self, model, images, targets, **kwargs) -> torch.Tensor:
        """
        Args:
            model: Victim model
            images: (B, C, H, W) normalized [0, 1]
            targets: Batch dict for loss computation
        """
        clean_images = images.clone().detach()
        B, C, H, W = images.shape
        device = images.device
        
        # --- Step 1: Saliency Extraction ---
        # Enable grad on images to get saliency
        images_req_grad = images.clone().detach().requires_grad_(True)
        
        preds = model(images_req_grad)
        loss_dict = model.compute_loss(preds, targets)
        loss = loss_dict['loss']
        
        grad = torch.autograd.grad(loss, images_req_grad, retain_graph=False, create_graph=False)[0]
        # Saliency: max absolute gradient across channels -> (B, 1, H, W)
        saliency = torch.max(torch.abs(grad), dim=1, keepdim=True)[0]
        
        # Initialize Global Density Map
        global_density_mask = torch.zeros((B, 1, H, W), device=device)
        
        gt_bboxes = targets.get('bboxes', [])
        batch_idx = targets.get('batch_idx', [])
        
        # --- Step 2 & 3: Local Shape Sampling & Best-of-N Selection ---
        if len(gt_bboxes) > 0:
            for b_i in range(B):
                # Get targets for this image
                mask = (batch_idx == b_i)
                if not mask.any():
                    continue
                
                img_gt = gt_bboxes[mask] # normalized cx, cy, w, h
                
                # Convert to pixel coords xyxy
                img_gt_xyxy = xywh2xyxy(img_gt)
                img_gt_xyxy[:, [0, 2]] *= W
                img_gt_xyxy[:, [1, 3]] *= H
                
                # Ensure correct device
                if img_gt_xyxy.device != device:
                    img_gt_xyxy = img_gt_xyxy.to(device)
                
                # Process each target
                for box in img_gt_xyxy:
                    x1, y1, x2, y2 = box.int()
                    
                    # Add Context Padding
                    pad = self.context_pad
                    x1 = max(0, x1.item() - pad)
                    y1 = max(0, y1.item() - pad)
                    x2 = min(W, x2.item() + pad)
                    y2 = min(H, y2.item() + pad)
                    
                    h_box, w_box = y2 - y1, x2 - x1
                    if h_box <= 0 or w_box <= 0:
                        continue
                        
                    # Extract Local Saliency
                    local_saliency = saliency[b_i, 0, y1:y2, x1:x2] # (h, w)
                    
                    # Generate N Candidate Shapes
                    # (N, 1, h, w)
                    candidates = self._generate_continuous_shape(h_box, w_box, device)
                    
                    # Calculate Coverage Score
                    # sum(saliency * candidate)
                    # broadcast local_saliency to (N, 1, h, w)
                    scores = (candidates * local_saliency.unsqueeze(0).unsqueeze(0)).sum(dim=(2, 3)).squeeze()
                    
                    # Select Best Shape
                    if self.N > 1:
                        best_idx = torch.argmax(scores)
                        best_mask = candidates[best_idx, 0] # (h, w)
                    else:
                        best_mask = candidates[0, 0]
                        
                    # --- Step 4: Global Max-Composition ---
                    # Update global mask using Max operation
                    current_global_crop = global_density_mask[b_i, 0, y1:y2, x1:x2]
                    global_density_mask[b_i, 0, y1:y2, x1:x2] = torch.max(current_global_crop, best_mask)

        # --- Step 5: Physical Rendering ---
        # Edge Feathering (Gaussian Blur)
        # Note: global_density_mask is binary-ish (0 or 1), feathering makes it smooth float
        feathered_density = self.feather_blur(global_density_mask)
        
        # Apply Atmospheric Scattering
        # I_adv = I * exp(-beta * d) + A * (1 - exp(-beta * d))
        # d here is the feathered density mask.
        # But wait, density usually represents optical depth.
        # If the mask is 0-1, 'beta' controls the max thickness.
        
        density = feathered_density * self.beta  # Scale by beta
        transmission = torch.exp(-density)
        
        adv_images = clean_images * transmission + self.A * (1 - transmission)
        adv_images = torch.clamp(adv_images, 0, 1)
        
        if kwargs.get('return_transmission', False):
            return adv_images, transmission

        return adv_images
