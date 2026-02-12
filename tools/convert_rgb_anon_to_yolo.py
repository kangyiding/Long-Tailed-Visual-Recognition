import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def convert_rgb_anon_to_yolo():
    # Configuration
    json_path = '/root/autodl-tmp/dky/foggy_test/data/rgb_anon/gt_detection/fog/instancesonly_fog_train_gt_detection.json'
    img_root = '/root/autodl-tmp/dky/foggy_test/data/rgb_anon' # JSON file_name is relative to this
    output_dir = '/root/autodl-tmp/dky/foggy_test/data/rgb_anon_yolo'
    
    # Category mapping (RGB Anon ID -> YOLO ID)
    cat_mapping = {
        24: 0, # person
        25: 1, # rider
        26: 2, # car
        27: 3, # truck
        28: 4, # bus
        31: 5, # train
        32: 6, # motorcycle
        33: 7  # bicycle
    }
    
    # Create directories
    images_dir = Path(output_dir) / 'images' / 'val'
    labels_dir = Path(output_dir) / 'labels' / 'val'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading annotations from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Index images by id
    images = {img['id']: img for img in data['images']}
    
    # Group annotations by image_id
    anns_by_img = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in anns_by_img:
            anns_by_img[img_id] = []
        anns_by_img[img_id].append(ann)
        
    print(f"Converting {len(images)} images...")
    
    for img_id, img_info in tqdm(images.items()):
        file_name = img_info['file_name'] # e.g. fog/train/GP010475/GP010475_frame_001043_rgb_anon.png
        
        # Original image path
        src_img_path = Path(img_root) / file_name
        if not src_img_path.exists():
            print(f"Warning: Image not found: {src_img_path}")
            continue
            
        # Target image path (flattened)
        dst_img_name = os.path.basename(file_name)
        dst_img_path = images_dir / dst_img_name
        
        # Create symlink if not exists
        if not dst_img_path.exists():
            try:
                os.symlink(src_img_path, dst_img_path)
            except OSError as e:
                print(f"Error linking {src_img_path} to {dst_img_path}: {e}")
                continue
        
        # Create label file
        label_file = labels_dir / (Path(dst_img_name).stem + '.txt')
        
        img_w = img_info['width']
        img_h = img_info['height']
        
        yolo_anns = []
        if img_id in anns_by_img:
            for ann in anns_by_img[img_id]:
                cat_id = ann['category_id']
                if cat_id not in cat_mapping:
                    continue # Skip unknown categories
                
                cls_id = cat_mapping[cat_id]
                bbox = ann['bbox'] # x, y, w, h
                
                # Convert to YOLO format
                x, y, w, h = bbox
                
                # Normalize
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                
                # Clip to [0, 1]
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w_norm = max(0.0, min(1.0, w_norm))
                h_norm = max(0.0, min(1.0, h_norm))
                
                yolo_anns.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                
        # Write label file
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_anns))
            
    print("Conversion complete!")
    print(f"Validation images: {images_dir}")
    print(f"Validation labels: {labels_dir}")

if __name__ == "__main__":
    convert_rgb_anon_to_yolo()
