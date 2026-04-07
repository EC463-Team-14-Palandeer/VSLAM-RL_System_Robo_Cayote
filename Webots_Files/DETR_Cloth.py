import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import pandas as p
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou

from tqdm import tqdm # <-- The Holy Progress Bar library 

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.amp import GradScaler, autocast

DATASET_DIR = "YOLO_Dataset" # Changed the name of Dataset
SAVE_DIR = "DETR_Training_Results_System_Test"
EPOCHS = 60 # Did 10 for a short test. Do more for real thing. :) --> 400 Epochs for full test?
BATCH_SIZE = 32 # Doing this test on an H200, so it should be good to handle 64 Batch size, right?
DEVICE = None 
ISDAVIDGROSS = True
CLASS_NUM = 13 # The Deep Fashion dataset has about 13 articles of clothing (long sleeves, short sleeves, etc.). 

print("Starting Pre-Training Procedure...")

if(torch.cuda.is_available()):
    print(f"CUDA Exists! Using Device: {torch.cuda.get_device_name()}")
    DEVICE = torch.device(0) # Use for GPU
else: 
    print("CUDA Does not exist! Need to use CPU...")
    DEVICE = 'cpu' # Use for CPU



class YoloDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))] # This dataset uses .jpg files.
    
    def __len__(self):
        return len(self.img_names)
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        label_name = img_name.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x, y, w, h = map(float, line.strip().split())
                    labels.append(int(class_id))
                    boxes.append([x, y, w, h])
        
        target = {
            'labels': torch.tensor(labels, dtype=torch.int64),
            'boxes': torch.tensor(boxes, dtype=torch.float32)
        }

        if self.transform:
            image = self.transform(image)
        
        return (image, target)
    
def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return (images, targets)


# We'll likely be doing RT-DETR v1 or v2, not regular DETR (Appareantly it kind of sucks on its own :P).
# DETR Paper:       https://arxiv.org/pdf/2005.12872 <-- Cool concept, not good in real usage. (Really interesting read, though)
# RT-DETR Paper:    https://arxiv.org/pdf/2304.08069 <-- Supposedly these guys faked their results (Results are very specific to dataset and optimized towards it; hard to replicate)
# RT-DETR v2 Paper: https://arxiv.org/pdf/2407.17140 <-- We're going with this one!


# Most papers define the number of queries to be ideal at 100 --> ADJUST AS NEEDED
class DETR_Deer(nn.Module):
    def __init__(self, num_classes=CLASS_NUM, num_queries=100, hidden_dim=256):
        super(DETR_Deer, self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = 0.1
        self.register_buffer('empty_weight', empty_weight)
        
        # The backbone of DETR is usually ResNet-50 or ResNet-101:
        resnet_preproc = models.resnet50(weights='DEFAULT') # Use pretraiend here to save time --> At least for now...
        backbone_layer = list(resnet_preproc.children())[:-2] # --> Get rid of Mean Pooling and Fully Connected Layers at the end >:)
        self.backbone = nn.Sequential(*backbone_layer)
        self.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=6)

        self.pos_enc = nn.Parameter(torch.randn(400, hidden_dim))
        self.query_position = nn.Parameter(torch.randn(num_queries, hidden_dim))
        
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=6)
        
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1) # Don't forget the "No Object Class" (That's why +1)
        #self.bbox_embed = nn.Linear(hidden_dim, 4) # CenterX, CenterY, Width, Length
        self.bbox_embed = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 4)) # 3 Layer MLP
        
        

    def forward(self, x):
        feature = self.backbone(x)
        temp = self.input_proj(feature)
        batch_size = x.shape[0]
        temp = temp.flatten(2).permute(0, 2, 1) 
        


        position = self.pos_enc.unsqueeze(0)
        memory = self.transformer_encoder(temp + position)
        queries = self.query_position.unsqueeze(0).repeat(batch_size, 1, 1)
        temp2 = self.transformer_decoder(queries, memory) # Give the decoder both the queries to look for and the actual embeddings.

        output_class = self.class_embed(temp2)
        # Use the sigmoid function for coordinates, since it can keep things in between 0 and 1 --> Need percentages :)
        # Reminder that Sigmoid is: 1/(1 + e^-x)
        # Hi David!
        output_coordinate = self.bbox_embed(temp2).sigmoid()
        return output_class, output_coordinate


def box_cxcywh_toxyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


# |    | l
# ||   |__

# Loss function uses the Hungarian Algorithm to essentially only reward/punish and learn from the boxes that are actually
# near the object of desire, or have the right class, and not those who are completely off.
# Also --> Make sure that you don't give too much reward for it seeing "Nothing" too often. --> Can't be same as seeing something.
class DETR_Loss(nn.Module):
    def __init__(self, num_classes=CLASS_NUM):
        super(DETR_Loss, self).__init__()
        self.num_classes = num_classes
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = 0.05 # <--- Used to be 0.1; Changed to 0.05 to punish the safe "None" guessing on everything.
        self.register_buffer('empty_weight', empty_weight)

    def forward(self, predicted_class, predicted_bbox, targets):
        batch_size = predicted_class.shape[0]
        total_class_loss = 0
        total_bbox_loss = 0
        num_actual_targets = 0

        for i in range(batch_size):
            out_prob = predicted_class[i].softmax(-1)
            out_bbox = predicted_bbox[i]
            target_ids = targets[i]['labels']
            target_bbox = targets[i]['boxes']

            num_actual_targets += len(target_ids)

            if len(target_ids) == 0:
                continue
            cost_class = -out_prob[:, target_ids]
            cost_bbox = torch.cdist(out_bbox, target_bbox, p=1)

            C = cost_class + cost_bbox
            C_numpy = C.cpu().detach().numpy()
            pred_indices, target_indices = linear_sum_assignment(C_numpy)

            # Only grab those that are close to the actual object:
            matched_prediction_boxes = out_bbox[pred_indices]
            matched_target_boxes = target_bbox[target_indices]
            matched_target_ids = target_ids[target_indices]

            
            predicted_xyxy = box_cxcywh_toxyxy(matched_prediction_boxes)
            target_xyxy = box_cxcywh_toxyxy(matched_target_boxes)
            bbox_loss = F.l1_loss(matched_prediction_boxes, matched_target_boxes)
            giou_loss = 1 - torch.diag(generalized_box_iou(predicted_xyxy, target_xyxy))
            #total_bbox_loss += (bbox_loss + giou_loss.mean())
            total_bbox_loss += (5.0 * bbox_loss + 2.0 * giou_loss.mean()) # Changes the ratio of the giou and L1 losses so that the L1 has a bit more authority.
            
            
            target_classes = torch.full(predicted_class[i].shape[:1], self.num_classes, dtype=torch.int64, device=predicted_class.device)
            target_classes[pred_indices] = matched_target_ids
            class_loss = F.cross_entropy(predicted_class[i], target_classes, weight=self.empty_weight)
            total_class_loss += class_loss

        if(num_actual_targets == 0):
            num_actual_targets = 1 # Can't divide by 0 :P
        #return (total_class_loss/batch_size) + (total_bbox_loss/batch_size)
        return ((total_class_loss + total_bbox_loss) / num_actual_targets)



def generate_plot(x_coordinate, y_coordinate, x_label, y_label, plot_title, plot_reason):
    # Plot things like loss, learning rate, mAP, etc. to see how DETR is performing. All data will get saved into a 
    # file, and then use that to generate the plot and save it using savefig().
    os.makedirs(SAVE_DIR, exist_ok=True)

    plt.figure()
    plt.plot(x_coordinate, y_coordinate)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)

    filepath = os.path.join(SAVE_DIR, f"{plot_reason}_figure.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot {filepath}")

    return

def generate_conf_mat(ground_truth, predicted_results):
    # Generate a confusion matrix of the predicted vs true results to get a good gague on the model's performance.
    os.makedirs(SAVE_DIR, exist_ok=True)
    conf_mat = confusion_matrix(ground_truth, predicted_results)
    labels = ["Short Sleeve Top", "Long Sleeve Top", "Short Sleeve Outwear", "Long Sleeve Outwear", "Vest", "Sling", "Shorts", "Trousers", "Skirt", "Short Sleeve Dress", "Long Sleeve Dress", "Vest Dress", "Sling Dress", "Background"] # We will be doing clothing discriminiation. This a larger dataset, so we'll get to see how DETR performs with a larger dataset against YOLO.


    fig, ax = plt.subplots(figsize=(12, 12)) # Change the size of the matrix to fit all things! :)
    conf_mat_show = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=labels)
    conf_mat_show.plot(cmap='Blues', ax=ax)

    filepath = os.path.join(SAVE_DIR, "Confusion_Matrix_DETR.png")
    plt.savefig(filepath)
    plt.close(fig)
    print(f"Saved Confusion Matrix to {filepath}")
    return

def save_train_data(total_train_time, total_epoch_num, batch_size, device, final_avg_loss):
    os.makedirs(SAVE_DIR, exist_ok=True)
    summary_path = os.path.join(SAVE_DIR, "training_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("DETR Training Summary:\n")
        f.write(f"Total Train Time: {total_train_time/60:.2f} Minutes\n")
        f.write(f"Total Epoch Number: {total_epoch_num}\n")
        f.write(f"Device Used for Training: {device}\n")
        f.write(f"Batch Size for Training: {batch_size}\n")
        f.write(f"Final Average Loss: {final_avg_loss[-1]:.4f}\n")
    print(f"Saved Summary Data to {summary_path}")
    
    return




def main():

    train_image_dir = os.path.join(DATASET_DIR, "images", "train")
    train_label_dir = os.path.join(DATASET_DIR, "labels", "train")
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_image_dir = os.path.join(DATASET_DIR, "images", "validation")
    val_label_dir = os.path.join(DATASET_DIR, "labels", "validation")



    
    train_dataset = YoloDataset(train_image_dir, train_label_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, num_workers=16, pin_memory=True)
    
    val_dataset = YoloDataset(val_image_dir, val_label_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn, num_workers=16, pin_memory=True)
    
    model = DETR_Deer(num_classes=CLASS_NUM).to(DEVICE)
    model = torch.compile(model) # Make it so it works really well for H200 GPU
    critic = DETR_Loss(num_classes=CLASS_NUM).to(DEVICE)

    parameter_dicts = [{"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]}, 
                       {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": 1e-5}]
    optimizer = torch.optim.AdamW(parameter_dicts, lr=5e-5)  # Changed from learning rate 1e-4 --> 5e-5

    epochs = EPOCHS
    epoch_losses = []

    print(f"Started Training on: {DEVICE}")
    start_time = time.time() # Get starting time to record total training time later.
    best_loss = float('inf')
    

    scalar = GradScaler('cuda')


    model.train()
    for epoch in range(epochs):
        total_loss_for_epoch = 0
        
        # Bring out the holy bar!
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for images, targets in progress_bar:
            images = images.to(DEVICE)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with autocast('cuda'):
                predicted_class, predicted_bbox = model(images)
                loss = critic(predicted_class, predicted_bbox, targets)
            #loss.backward()
            #optimizer.step()
            
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            scalar.update()


            total_loss_for_epoch += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss_for_epoch/len(train_loader)
        epoch_losses.append(avg_loss)

        # Save the best model --> We can later compare it to the latest saved model to see if it actually outperforms it.
        # Only save dicts though, don't save the entire thing.
        if (avg_loss < best_loss):
            best_loss = avg_loss
            best_model_path = os.path.join(SAVE_DIR, "DETR_Best_System_Test.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New Best model found --> Saved it to {best_model_path} Loss: {best_loss}")


        # Also save a model every 5 epochs (And the last epoch as well):
        if ((epoch + 1) % 5 == 0) or ((epoch + 1) == EPOCHS):
            interval_model_path = os.path.join(SAVE_DIR, f"DETR_Epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), interval_model_path)
            print(f"\n\nSaved the newest of the interval models! Saved to {interval_model_path}")







        end_time = time.time()
        total_train_time = end_time - start_time


    generate_plot(x_coordinate=range(1, epochs + 1), y_coordinate=epoch_losses, x_label="Epoch", y_label="Average Loss", plot_title="DETR Training Loss over Time:", plot_reason="Training_Loss")
    print("Training Complete! Running evaluation for mAP...")
    model.eval()
    
    metric = MeanAveragePrecision(box_format='cxcywh', iou_type='bbox') # Initialize the mAP 



    #all_ground_truths = []
    #all_predictions = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating mAP"):
            images = images.to(DEVICE)
            with autocast('cuda'):
                predicted_class, predicted_bbox = model(images)

                #predicted_class, _ = model(images)
            

            batch_size_current = images.shape[0]
            preds_list = []
            actual_targets_list = []

            for i in range(batch_size_current):
                probs = predicted_class[i].softmax(-1)
                max_probs, pred_classes = probs.max(dim=-1)

                threshold = 0.5
                keep_indices = (max_probs > threshold) & (pred_classes != CLASS_NUM)

                valid_boxes = predicted_bbox[i][keep_indices]
                valid_scores = max_probs[keep_indices]
                valid_labels = pred_classes[keep_indices]

                preds_list.append(dict(boxes=valid_boxes, scores=valid_scores, labels=valid_labels))
                actual_targets_list.append(dict(boxes=targets[i]['boxes'].to(DEVICE), labels=targets[i]['labels'].to(DEVICE)))

            metric.update(preds_list, actual_targets_list)




           # probs = predicted_class[0].softmax(-1)
                #top_classes = probs.argmax(-1)
                #model_found_human = 1 if 0 in top_classes else 0
            #max_probs, pred_classes = probs.max(dim=-1)

            #threshold = 0.5 # Make a 50% confidence threshold for this...
            
            #keep_indices = (max_probs > threshold) & (pred_classes != CLASS_NUM)


            #valid_boxes = predicted_bbox[0][keep_indices]
            #valid_scores = max_probs[keep_indices]
            #valid_labels = pred_classes[keep_indices]


           # if len(valid_predict) > 0:
            #    # Grab most predicted:  
             #   image_predict = torch.mode(valid_predict).values.item()
            #else:
             #   image_predict = CLASS_NUM # If nothing detected as much, then just assign it to background noise and don't display.

            #preds = [dict(boxes=valid_boxes, scores=valid_scores, labels=valid_labels)]
            #actual_targets = [dict(boxes=targets[0]['boxes'].to(DEVICE), labels=targets[0]['labels'].to(DEVICE))]

            #metric.update(preds, actual_targets)
        map_results = metric.compute() # Compute final mAP across entire validation
        print("\nDETR Evaluation Results: ")
        print(f"mAP (IoU=0.5:0.95): {map_results['map']}")
        print(f"mAP (IoU=0.5): {map_results['map_50']}")
        print(f"mAP (IoU=0.75): {map_results['map_75']}")



        #    actual_labels = targets[0]['labels'] if 'labels' in targets[0] else []
        #    if len(actual_labels) > 0:
        #        image_actual = actual_labels[0].item()
        #    else:
        #        image_actual = CLASS_NUM
        
        #    all_predictions.append(image_predict)
        #    all_ground_truths.append(image_actual)



            # Check ground truth: Did the target actually have a human label (0)?
            #actual_labels = targets[0]['labels'] if 'labels' in targets[0] else []
            #actual_has_human = 1 if 0 in actual_labels else 0
            
            # Invert to match your labels ["Human" (0), "Not Human" (1)]
            #all_predictions.append(0 if model_found_human else 1)
            #all_ground_truths.append(0 if actual_has_human else 1)

    #try:
    #    generate_conf_mat(all_ground_truths, all_predictions)
    #except Exception as e:
    #    print(f"Failed to generate Confusion Matrix, due to {e}")
    try:
        save_train_data(total_train_time, epochs, BATCH_SIZE, str(DEVICE), epoch_losses)
    except Exception as e:
        print(f"Failed to save training data, due to {e}")

    model_path = os.path.join(SAVE_DIR, "DETR_Clothes_model_weight.pth")
    model_path_full = os.path.join(SAVE_DIR, "DETR_Clothes_model_full.pth")
    torch.save(model.state_dict(), model_path)
    torch.save(model, model_path_full)

    print(f"Saved the weights and full models to {model_path} and {model_path_full}")

if (__name__ == "__main__") and (ISDAVIDGROSS == True):
    print("Everything loaded and works...")
    print("Starting Training!")
    print("Let's hope this works...")
    main()
