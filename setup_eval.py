import os
import shutil
import glob
from pathlib import Path

def setup_eval_dataset():
    # Config
    attack_dir = Path("runs/attack/attention_fog_grad")
    label_root = Path("data/OpenDataLab___CityScapes/labels/train")
    eval_root = Path("runs/eval_temp")
    
    # Clean up
    if eval_root.exists():
        shutil.rmtree(eval_root)
    
    img_dir = eval_root / "images"
    lbl_dir = eval_root / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    
    # Find generated images
    # They are in city subfolders, e.g., aachen/xxx.png
    image_files = list(attack_dir.rglob("*.png"))
    print(f"Found {len(image_files)} images to evaluate.")
    
    for img_path in image_files:
        # Copy image (flatten structure for simplicity or keep it?)
        # YOLO works fine with flat structure in val
        dest_img_path = img_dir / img_path.name
        shutil.copy(img_path, dest_img_path)
        
        # Find label
        # Label path: label_root / city / filename.txt
        # We need to extract city name from img_path parent? 
        # Or just search in label_root?
        # CityScapes structure: aachen/aachen_xxx.png
        city = img_path.parent.name
        label_name = img_path.stem + ".txt"
        src_label_path = label_root / city / label_name
        
        if src_label_path.exists():
            shutil.copy(src_label_path, lbl_dir / label_name)
        else:
            print(f"Warning: Label not found for {img_path.name}")

    # Create YAML
    yaml_content = f"""
path: {eval_root.absolute()}
train: images
val: images
test: images
names:
  0: person
  1: rider
  2: car
  3: truck
  4: bus
  5: train
  6: motorcycle
  7: bicycle
"""
    with open("eval_temp.yaml", "w") as f:
        f.write(yaml_content)
    
    print("Evaluation dataset setup complete.")

if __name__ == "__main__":
    setup_eval_dataset()
