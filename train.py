import argparse
import os
import torch
import torch.optim as optim
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg

from core import MODELS
from pipelines.trainer import Trainer
# Import yolo_wrapper to register the model
import models.yolo_wrapper

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv8 model')
    parser.add_argument('--model', type=str, default='yolov8m', help='Model variant (e.g., yolov8n, yolov8m)')
    parser.add_argument('--data', type=str, default='cityscapes.yaml', help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--imgsz', type=int, default=1024, help='Image size for training')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Create Save Directory
    # Moved to inside loop to include dataset name, or we can keep it here as base
    # save_dir = os.path.join(args.save_dir, args.model)
    # os.makedirs(save_dir, exist_ok=True)
    # print(f"Checkpoints will be saved to: {save_dir}")

    # 1. Load Data Config
    data_cfg = check_det_dataset(args.data)
    num_classes = data_cfg.get('nc')
    class_names = data_cfg.get('names')
    print(f"Dataset has {num_classes} classes.")
    
    cfg = get_cfg(overrides={'data': args.data, 'mode': 'train', 'imgsz': args.imgsz})
    
    # 2. Build Model
    # Use Registry to build model
    print(f"Initializing {args.model}...")
    model = MODELS.build(cfg={
        'type': 'yolov8', 
        'model_path': f"{args.model}.pt",
        'device': args.device,
        'num_classes': num_classes,
        'names': class_names
    })

    # Use Ultralytics to build dataloader
    # Note: build_dataloader usually requires a dataset object or path
    # Let's keep it simple: use the internal YOLO mechanism to get the dataloader if possible
    # Or manually build it.
    # For now, let's use the standard Ultralytics dataset builder
    try:
        # This is a bit complex to hook into without using the full YOLO engine.
        # Alternative: Use the model.yolo.train() for data loading setup but run our loop?
        # Or just use standard PyTorch dataloader if we had a Dataset class.
        # Let's try to reuse Ultralytics dataset.
        train_path = data_cfg['train']
        
        # This might be tricky without full config context. 
        # Simpler approach for this task: Use a dummy dataloader or try to instantiate the real one.
        # Let's assume the user has coco128.yaml available or it will download.
        dataset = build_yolo_dataset(cfg, train_path, batch=args.batch_size, data=data_cfg, mode='train', rect=False, stride=32)
        dataloader = build_dataloader(dataset, batch=args.batch_size, workers=4, shuffle=True, rank=-1)
        print("DataLoader built successfully.")
    except Exception as e:
        print(f"Error building dataloader: {e}")
        print("Falling back to dummy dataloader for demonstration if needed, but stopping here.")
        return

    # 4. Optimizer
    # Access the underlying PyTorch model parameters
    # Improved optimizer setup: Group parameters
    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, torch.nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    if not g0 and not g1 and not g2:
         # Fallback if manual grouping fails (e.g. wrapper structure)
         print("Warning: Parameter grouping failed, using default all-parameters optimizer.")
         optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0005)
    else:
        # Use AdamW like Ultralytics auto-optimizer often selects for small batches/transfer
        optimizer = optim.AdamW(g0, lr=args.lr, betas=(0.9, 0.999)) # BN weights (no decay)
        optimizer.add_param_group({'params': g1, 'weight_decay': 0.0005})  # weights (with decay)
        optimizer.add_param_group({'params': g2, 'weight_decay': 0.0})  # biases (no decay)
        print(f"Optimizer groups: {len(g0)} BN weights, {len(g1)} weights, {len(g2)} biases")

    # 5. Scheduler
    # Linear decay from lr0 to lr0 * lrf (default lrf=0.01)
    lrf = 0.01
    lf = lambda x: (1 - x / args.epochs) * (1.0 - lrf) + lrf
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 6. Trainer
    trainer = Trainer(model, optimizer, scheduler=scheduler, device=args.device)

    # 7. Training Loop
    print("Starting training...")
    best_map = 0.0
    
    # Construct save directory with dataset name
    dataset_name = os.path.splitext(os.path.basename(args.data))[0]
    save_dir = os.path.join(args.save_dir, f"{dataset_name}_{args.model}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {save_dir}")

    for epoch in range(args.epochs):
        # Train one epoch
        metrics = trainer.train_epoch(dataloader, epoch=epoch)
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {metrics['train_loss']:.4f}")
        
        # Step Scheduler
        if scheduler:
            scheduler.step()
            
        # Validation
        val_metrics = trainer.validate(args.data, batch_size=args.batch_size, imgsz=args.imgsz)
        current_map = val_metrics.get("mAP50-95", 0.0)
        print(f"Epoch {epoch+1}/{args.epochs} - mAP50: {val_metrics.get('mAP50', 0):.4f} - mAP50-95: {current_map:.4f}")
        
        # Save Best Checkpoint
        if current_map > best_map:
            best_map = current_map
            save_path = os.path.join(save_dir, "best.pt")
            # Save in Ultralytics-compatible format (saving the model object directly)
            # This allows YOLO(save_path) to work
            try:
                # Ensure model is in eval mode before saving? Not strictly necessary but good practice.
                # We need to save the underlying DetectionModel, not the Wrapper
                # Note: We should probably move to CPU if we want portability, but for now let's keep it simple.
                torch.save({
                    'epoch': epoch,
                    'best_fitness': best_map,
                    'model': model.pytorch_model, 
                    'optimizer': optimizer.state_dict(),
                    'train_args': getattr(model.yolo, 'overrides', {}),  # Save args if available
                }, save_path)
                print(f"New best model saved to {save_path} (mAP: {best_map:.4f})")
            except Exception as e:
                print(f"Failed to save best checkpoint: {e}")
        
        # Save Latest Checkpoint (optional, overwrites each epoch)
        latest_path = os.path.join(save_dir, "last.pt")
        try:
            torch.save({
                'epoch': epoch,
                'model': model.pytorch_model,
                'optimizer': optimizer.state_dict(),
                'train_args': getattr(model.yolo, 'overrides', {}),
            }, latest_path)
        except Exception as e:
            print(f"Failed to save latest checkpoint: {e}")

if __name__ == "__main__":
    main()
