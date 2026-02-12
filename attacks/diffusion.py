import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, List
from PIL import Image, ImageFilter
from torchvision.transforms import ToPILImage, ToTensor
from core.registry import ATTACKERS
from core.abstract_attacker import BaseAttacker
from core.abstract_model import BaseModel

try:
    from diffusers import StableDiffusionImg2ImgPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

@ATTACKERS.register_module("diffusion_fog")
class DiffusionFogAttacker(BaseAttacker):
    def __init__(self, 
                 model_id: str = "runwayml/stable-diffusion-v1-5", 
                 prompt: str = "dense white fog, mist, smoke, whiteout, low visibility",
                 negative_prompt: str = "clear, sharp, sunny, bright, high contrast, objects, details",
                 strength: float = 0.7,  # Increased strength for dense fog source
                 seed: int = 42,         # Random seed
                 device: str = "cuda",
                 offload_cpu: bool = False, # Option to offload to CPU to save VRAM
                 **kwargs):
        super().__init__(**kwargs)
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Please install diffusers: pip install diffusers transformers accelerate")
        
        self.device = device
        self.model_id = model_id
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.strength = strength
        self.seed = seed
        self.offload_cpu = offload_cpu
        self.pipe = None  # Lazy loading

    def _load_pipeline(self):
        if self.pipe is None:
            print(f"Loading Diffusion Model: {self.model_id}...")
            # Use fp16 for faster generation and lower memory
            dtype = torch.float16 if "cuda" in self.device else torch.float32
            
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id, 
                torch_dtype=dtype,
                safety_checker=None, # Disable safety checker to prevent black images
                requires_safety_checker=False
            )
            
            # VRAM Optimization
            if self.offload_cpu and "cuda" in self.device:
                # Requires 'accelerate' library
                try:
                    self.pipe.enable_model_cpu_offload()
                    print("Enabled model CPU offloading for Stable Diffusion.")
                except AttributeError:
                    print("Warning: accelerate not installed or version too old. Falling back to .to(device).")
                    self.pipe.to(self.device)
            else:
                self.pipe.to(self.device)
            
            # Disable progress bar to keep logs clean during attack loop
            self.pipe.set_progress_bar_config(disable=True)

    def _generate_density_map(self, images: torch.Tensor, targets: Any) -> torch.Tensor:
        """
        Generate a physically plausible fog density map.
        Components:
        1. Vertical Gradient: Simulates atmospheric depth (sky is foggier than ground).
        2. Perlin-like Noise: Simulates random volumetric variations.
        3. Target-Aware Blobs: Increases density around targets (as requested) but using smooth Gaussian falloff.
        """
        B, C, H, W = images.shape
        device = images.device
        
        # 1. Vertical Gradient (Depth Heuristic)
        # 0.1 at bottom -> 0.5 at top
        y = torch.linspace(0, 1, steps=H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        gradient_map = 0.1 + (y * 0.4)
        
        # 2. Smooth Noise (Simulate Volumetric Heterogeneity)
        # Generate low-res noise and upsample for smooth "cloud" look
        noise_low = torch.rand(B, 1, H // 32, W // 32, device=device)
        noise_map = F.interpolate(noise_low, size=(H, W), mode='bicubic', align_corners=False)
        noise_map = (noise_map - 0.5) * 0.2 # Variation +/- 0.1
        
        density_map = gradient_map + noise_map
        
        # 3. Target-Aware Gaussian Blobs
        if isinstance(targets, dict) and 'bboxes' in targets:
            bboxes = targets['bboxes']
            batch_idx = targets.get('batch_idx')
            
            if bboxes is not None and len(bboxes) > 0:
                # Create coordinate grid
                yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
                
                # Process each image in batch
                for i in range(B):
                    if batch_idx is not None:
                        idx_mask = (batch_idx == i)
                        img_boxes = bboxes[idx_mask]
                    else:
                        img_boxes = bboxes if B == 1 else []
                        
                    for box in img_boxes:
                        # Handle normalized/unnormalized coordinates
                        box_np = box.detach().cpu().numpy() # Use numpy for coordinate check logic if needed, but torch is faster for math
                        
                        # Assuming normalized xywh if max value is small
                        if torch.max(box) <= 1.5:
                            cx, cy, w_box, h_box = box
                            cx = cx * W
                            cy = cy * H
                            w_box = w_box * W
                            h_box = h_box * H
                        else:
                            # xywh unnormalized
                            x1, y1, w_box, h_box = box
                            cx = x1 + w_box / 2
                            cy = y1 + h_box / 2

                        # Gaussian parameters
                        # Sigma proportional to object size
                        sigma_x = w_box / 2.0
                        sigma_y = h_box / 2.0
                        
                        # Add Gaussian blob to density map
                        # density += amp * exp(-((x-cx)^2/2sx^2 + (y-cy)^2/2sy^2))
                        # Optimization: Compute only within a ROI to save time, or just full grid for simplicity on GPU
                        
                        # Full grid calculation (vectorized on GPU is fast enough for 640x640)
                        gaussian = torch.exp(-(((xx - cx)**2 / (2 * sigma_x**2)) + ((yy - cy)**2 / (2 * sigma_y**2))))
                        
                        # Add to density map (accumulate fog)
                        # Scale amplitude: Target regions get +0.6 density
                        density_map[i, 0] += gaussian * 0.6

        # Clamp density to valid range [0, 1]
        return torch.clamp(density_map, 0.0, 1.0)

    def generate(self, model: BaseModel, images: torch.Tensor, targets: Any, **kwargs) -> torch.Tensor:
        """
        Two-stage generation:
        1. Physics-based Fog Simulation: Apply atmospheric scattering model.
        2. Diffusion Refinement: Use img2img with low strength to add realistic texture and lighting artifacts.
        """
        self._load_pipeline()
        
        batch_size = images.shape[0]
        orig_h, orig_w = images.shape[2], images.shape[3]
        
        # --- Stage 1: Physical Fog Simulation ---
        # I(x) = J(x)t(x) + A(1-t(x))
        # Let density map be d(x). t(x) = exp(-beta * d(x))? 
        # Simplified: density_map is directly (1-t), i.e., opacity of the fog.
        # I_fog = I_clean * (1 - density) + A * density
        
        density_map = self._generate_density_map(images, targets)
        
        # Atmospheric Light (A): Usually white/gray [1.0, 1.0, 1.0]
        A = 1.0 
        
        # Apply equation
        physically_foggy_images = images * (1 - density_map) + A * density_map
        physically_foggy_images = torch.clamp(physically_foggy_images, 0, 1)

        # --- Stage 2: Diffusion Refinement ---
        # We pass the physically foggy image to SD to "hallucinate" realistic details (swirls, light diffusion)
        # using a LOW strength to preserve the structure we just created.
        
        to_pil = ToPILImage()
        to_tensor = ToTensor()
        
        # Preprocess for SD (Dimension adjustment)
        def preprocess_image(img_tensor):
            pil_img = to_pil(img_tensor.cpu())
            w, h = pil_img.size
            new_w = (w // 8) * 8
            new_h = (h // 8) * 8
            if new_w != w or new_h != h:
                pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
            return pil_img

        pil_inputs = [preprocess_image(img) for img in physically_foggy_images]
        
        # Reduce strength for refinement to avoid "smearing" (hallucinating new geometry)
        # Use a lower strength than the default 0.7 if we want to preserve the physical structure
        refinement_strength = 0.35 
        
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        
        with torch.no_grad():
            prompt_batch = [self.prompt] * batch_size
            negative_prompt_batch = [self.negative_prompt] * batch_size
            
            refined_pils = self.pipe(
                prompt=prompt_batch,
                negative_prompt=negative_prompt_batch,
                image=pil_inputs,
                strength=refinement_strength,
                guidance_scale=7.5,
                num_inference_steps=20,
                generator=generator,
                output_type="pil"
            ).images

        # --- Post-process ---
        final_images = []
        for pil_img in refined_pils:
            if pil_img.size != (orig_w, orig_h):
                pil_img = pil_img.resize((orig_w, orig_h), Image.BILINEAR)
            final_images.append(to_tensor(pil_img))

        return torch.stack(final_images).to(self.device)
