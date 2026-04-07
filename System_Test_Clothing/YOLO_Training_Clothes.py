import os

import sys

import subprocess

from ultralytics import YOLO

import torch

import argparse

import shutil



def parse_args():

    p = argparse.ArgumentParser()

    # CHANGED: Default points to the directory created by the converter script

    p.add_argument('--data', type=str, default='YOLO_Shirt_Dataset', help='path to dataset root or data.yaml')

    p.add_argument('--imgsz', type=int, default=512, help='image size (square)')

    p.add_argument('--batch', type=int, default=32, help='batch size (per GPU)')

    p.add_argument('--epochs', type=int, default=20, help='number of epochs')

    p.add_argument('--workers', type=int, default=8, help='dataloader workers')

    p.add_argument('--model', type=str, default='yolov8s.pt', help='which YOLOv8 variant or path to pretrained .pt')

    p.add_argument('--device', type=str, default='0', help='cuda device (like "0") or "cpu"')

    p.add_argument('--project', type=str, default='runs/train', help='save folder')

    p.add_argument('--name', type=str, default='exp', help='run name')

    return p.parse_args()



def main():

    args = parse_args()



    if args.device.lower() != 'cpu' and not torch.cuda.is_available():

        print("CUDA not available, switching to CPU.")

        args.device = 'cpu'



    DATA_YAML = args.data

    if not os.path.exists(DATA_YAML):

        raise FileNotFoundError(f"Dataset folder or data.yaml not found: {DATA_YAML}")



    # CHANGED: Updated directory structure to match standard YOLO and updated classes

    if os.path.isdir(DATA_YAML):

        root = os.path.abspath(DATA_YAML)

        

        data_yaml_text = f"""

train: {os.path.join(root, 'images', 'train')}

val:   {os.path.join(root, 'images', 'validation')}



# Classes

nc: 2

names: ['blue_shirt', 'red_shirt']

"""

        temp_yaml = os.path.join(root, 'data_from_script.yaml')

        print(f"Creating temporary data.yaml at {temp_yaml}")

        with open(temp_yaml, 'w') as f:

            f.write(data_yaml_text)

        DATA_YAML = temp_yaml



    print("Loading model:", args.model)

    model = YOLO(args.model)



    print("Starting training...")

    results = model.train(

        data=DATA_YAML,

        imgsz=args.imgsz,

        epochs=args.epochs,

        batch=args.batch,

        workers=args.workers,

        device=args.device,

        project=args.project,

        name=args.name,

        exist_ok=True

    )



    print("Training finished. Results summary:")

    print(results)



    best_pt = os.path.join(args.project, args.name, "weights", "best.pt")

    final_pt = os.path.join(args.project, args.name, "weights", "last.pt")

    print("Best checkpoint:", best_pt)

    

    eval_model = YOLO(best_pt if os.path.exists(best_pt) else model)

    print("Running validation...")

    val_results = eval_model.val(data=DATA_YAML, imgsz=args.imgsz, batch=args.batch,

                                 device=args.device, workers=args.workers)

    print("Validation results:")

    print(val_results)



    export_dir = os.path.join(args.project, args.name, "exported")

    os.makedirs(export_dir, exist_ok=True)

    try:

        print("Exporting to ONNX...")

        eval_model.export(format='onnx', imgsz=args.imgsz, opset=12, simplify=True, device=args.device)

    except Exception as e:

        print("ONNX export failed:", e)



    print("All done.")



if __name__ == '__main__':

    main()
