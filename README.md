# VSLAM-RL_System_Robo_Cayote
RL System for the Robo Cayote project done as part of the EC463/464 class (Senior Design). We are using a Soft Actor Critic approach to getting the robot to move and make decisions, and are using VSLAM to provide a costmap to the robot to allow for smarter decision-making.

# AI Systems:
## YOLO:
  For our YOLO model, we are using a YOLOv8s pre-trained model by Ultralytics. The reason for this is due to efficiency and training cost of training a YOLO model from scratch. YOLO performs a single pass for detection, and is as such very good for real-time robotics. With it, we have efficient detection for a normal circumstance. It uses anchor boxes to perform its detections, which makes it slightly innacurate, as it doesn't account for the size or distance of an object, which makes obscure images more complicated and erroneous to accurately detect. In our preliminary training with YOLOv8s (With a small-medium sized human-only dataset), we have noticed that YOLO has difficulties with blurry images or those with lots of noise in the background. Below is the Confusion Matrix for our preliminary model:
  <img width="1290" height="981" alt="Screenshot 2026-03-06 210840" src="https://github.com/user-attachments/assets/327f49dc-2831-4aad-b6b3-1f37587ba4fd" />
*<p align="center">Figure 1: YOLOv8s Preliminary Results</p>*

<p>As can be seen above in **Figure 1**, the model performed decently well (~60% accuracy overall), but when looked at closely, we notice it's innacuracies in capturing finer details or understanding the true nature of a human (or deer) anatomy.</p>

<img width="1124" height="330" alt="Screenshot 2025-12-02 152605" src="https://github.com/user-attachments/assets/9016b6bb-d3ff-4faf-b642-28ddbd88a2ad" />

*<p align="center">Figure 2: One of the resulting image detections for our Preliminary YOLOv8s</p>*


<p>While not a horrible case, it means that YOLOv8s is not entirely reliable for more complicated situations, which forces us to adapt further. Aside from further training and larger sets of data, this is inherently an issue caused by YOLO not regarding the amount of "zooming in" it performs when doing detection with its bounding boxes, as well as Non-Maximized Suppression (NMS) often leading to boxes that would be more helpful being deleted due to overlap.</p>



## DETR:
  We have decided to incorporate RT-DETR, a Transformer-based technique for real time detection into our model collection. DETR will act as a "checker" model, to verify if YOLO's uncertainty is correct or not.
  More often than not, YOLO struggles with detecting people in obscure images (For example: Far-away image of statue that is blurry), and as such often misclassifies something that isn't human as it being human.

  To combat this, we are implementing a "manager" code that will deterime if YOLO's confidence in an image is too low, and if it is, it pulls out DETR for a few seconds, to perform verification on whether YOLO is correct or not. 


## Reinforcement Learning:


# ROS Setup and Use Case:
