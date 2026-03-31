from ultralytics import YOLO

model = YOLO("human_yolo_final.pt") 

# Export directly to TensorRT engine format in FP16 (Half precision) --> This should make the inference run better and less computationally demanding
model.export(format="engine", half=True, workspace=4) # This will use up like 4GB of RAM so don't do anything else while optimizing this
