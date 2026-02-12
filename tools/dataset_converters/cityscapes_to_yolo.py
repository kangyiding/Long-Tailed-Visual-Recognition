"""
Cityscapes to YOLO format converter.
Reads gtFine JSONs and generates YOLO .txt labels.
"""
import os
import json
import glob
from pathlib import Path
from tqdm import tqdm
import argparse

# Cityscapes classes to YOLO classes mapping
# We use standard 8 classes for detection
CLASSES_MAP = {
    'person': 0,
    'rider': 1,
    'car': 2,
    'truck': 3,
    'bus': 4,
    'train': 5,
    'motorcycle': 6,
    'bicycle': 7
}

def convert_box(size, box):
    """
    Convert polygon bbox to YOLO format (normalized xywh).
    size: (width, height)
    box: (xmin, xmax, ymin, ymax)
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_cityscapes_to_yolo(root_path: str, output_path: str = None):
    """
    Convert Cityscapes gtFine to YOLO labels.
    
    Args:
        root_path: Path to Cityscapes root (containing gtFine and leftImg8bit).
        output_path: Path to save labels. If None, saves in 'labels' parallel to 'leftImg8bit'.
    """
    root = Path(root_path)
    gt_path = root / 'gtFine'
    
    if output_path is None:
        output_path = root / 'labels'
    else:
        output_path = Path(output_path)
    
    print(f"Searching for JSONs in {gt_path}...")
    # Recursive search for all polygons json files
    # pattern: gtFine/split/city/*_polygons.json
    json_files = sorted(list(gt_path.rglob('*_polygons.json')))
    
    print(f"Found {len(json_files)} JSON files. Starting conversion...")
    
    for json_file in tqdm(json_files):
        # Determine output path structure: labels/split/city/filename.txt
        # json_file: .../gtFine/train/aachen/aachen_000000_000019_gtFine_polygons.json
        # relative: train/aachen/aachen_000000_000019_gtFine_polygons.json
        rel_path = json_file.relative_to(gt_path)
        
        # New name: replace _gtFine_polygons.json with .txt
        txt_name = json_file.name.replace('_gtFine_polygons.json', '.txt')
        
        # Output directory: labels/train/aachen/
        save_dir = output_path / rel_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        txt_file = save_dir / txt_name
        
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        img_w = data['imgWidth']
        img_h = data['imgHeight']
        
        lines = []
        for obj in data['objects']:
            label = obj['label']
            if label in CLASSES_MAP:
                class_id = CLASSES_MAP[label]
                polygon = obj['polygon']
                
                # Extract bbox from polygon
                x_coords = [p[0] for p in polygon]
                y_coords = [p[1] for p in polygon]
                
                xmin = min(x_coords)
                xmax = max(x_coords)
                ymin = min(y_coords)
                ymax = max(y_coords)
                
                # Convert to YOLO
                b = (xmin, xmax, ymin, ymax)
                bb = convert_box((img_w, img_h), b)
                
                lines.append(f"{class_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")
        
        # Write to file (only if objects found? YOLO usually expects empty files for no objects)
        with open(txt_file, 'w') as f:
            f.writelines(lines)
            
    print(f"Conversion completed. Labels saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/OpenDataLab___CityScapes', help='Cityscapes root directory')
    args = parser.parse_args()
    
    convert_cityscapes_to_yolo(args.root)
