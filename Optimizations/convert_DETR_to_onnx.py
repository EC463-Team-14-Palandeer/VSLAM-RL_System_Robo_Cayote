import torch
import torch.onnx
from DETR import *
import sys
import os



model = DETR_Deer(num_classes=2) # Change to 3 for finalized model version
weights_path = "DETR_Best.pth" # Change as necessary
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model.load_state_dict(torch.load(weights_path, map_location=DEVICE))

try:
    model.eval() # NEED TO HAVE MODEL IN EVALUATION MODE
except Exception as e:
    print(f"Failed to start model in evaluation mode due to {e}. Unsafe to continue...\nEnding Code Runtime")
    sys.exit(1) # Exit code with code 1 (Failed to load)

dummy_input = torch.randn(1, 3, 640, 640) 
onnx_path = "DETR_Best.onnx"
print("Exporting .pth model to .onnx format!")

try:
    torch.onnx.export(
        model,                       # The model being exported
        dummy_input,                 # The fake image to trace
        onnx_path,                   # Where to save the file
        export_params=True,          # Store the trained weights inside the ONNX file
        opset_version=14,            # ONNX version (14 is highly compatible with Transformers)
        do_constant_folding=True,    # Optimizes constant operations for speed
        input_names=['images'],      # Name of the input layer
        output_names=['classes', 'boxes'] # Names of the output layers
    )
except Exception as f:
    print(f"Failed to run due to {f}! Ending Program!")
    sys.exit(2) # 2 --> Failed to Export error

print(f"Succeeded to export file! Congratulations! \nFile exported to: {os.curdir}")
sys.exit(0) # 0 --> Successful completion of code. 

