import os
import json
import cv2
import shutil
from tqdm import tqdm

DF2_ROOT = "DeepFashiom2/deepfashion2_original_images" # Update to your actual path
YOLO_ROOT = "YOLO_Dataset"

def convert_split(split_name):
    img_dir = os.path.join(DF2_ROOT, split_name, 'image')
    anno_dir = os.path.join(DF2_ROOT, split_name, 'annos')
    
    out_img_dir = os.path.join(YOLO_ROOT, 'images', split_name)
    out_lbl_dir = os.path.join(YOLO_ROOT, 'labels', split_name)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    
    if not os.path.exists(anno_dir):
        print(f"Skipping {split_name}, no annotations found.")
        return

    json_files = [f for f in os.listdir(anno_dir) if f.endswith('.json')]
    print(f"Converting {split_name} data...")
    
    for json_file in tqdm(json_files):
        with open(os.path.join(anno_dir, json_file), 'r') as f:
            data = json.load(f)
            
        img_name = json_file.replace('.json', '.jpg')
        img_path = os.path.join(img_dir, img_name)
        
        img = cv2.imread(img_path)
        if img is None: continue
        h, w, _ = img.shape
        
        yolo_labels = []
        
        for key, item in data.items():
            if not key.startswith('item'): continue
            
            # DeepFashion2 IDs are 1-13. YOLO/PyTorch expects 0-indexed.
            # So short sleeve top (1) becomes 0, long sleeve top (2) becomes 1.
            cat_id = int(item.get('category_id')) - 1 
            
            x_min, y_min, x_max, y_max = item.get('bounding_box')
            
            # Normalize for YOLO format
            x_center = ((x_min + x_max) / 2.0) / w
            y_center = ((y_min + y_max) / 2.0) / h
            box_w = (x_max - x_min) / w
            box_h = (y_max - y_min) / h
            
            yolo_labels.append(f"{cat_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
        
        if yolo_labels:
            shutil.copy(img_path, os.path.join(out_img_dir, img_name))
            with open(os.path.join(out_lbl_dir, json_file.replace('.json', '.txt')), 'w') as f:
                f.write('\n'.join(yolo_labels))

if __name__ == "__main__":
    convert_split('train')
    convert_split('validation')
    print("Dataset ready!")
