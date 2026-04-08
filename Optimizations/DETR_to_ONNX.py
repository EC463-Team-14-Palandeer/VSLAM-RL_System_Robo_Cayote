import torch
from DETR_Cloth import DETR_Deer

device = torch.device("cuda:0")
model = DETR_Deer(num_classes=2)
model.load_state_dict(torch.load("DETR_Best_System_Test.pth"))
model.to(device)
model.eval()

# Create a dummy image tensor of the correct size (1 image, 3 channels, 640x640)
dummy_input = torch.randn(1, 3, 640, 640).to(device)

print("Exporting to ONNX...")
torch.onnx.export(model, 
                  dummy_input, 
                  "DETR_best.onnx", 
                  export_params=True, 
                  opset_version=16, # Opset 16 works well for Transformers
                  input_names=['input'], 
                  output_names=['output_class', 'output_coordinate'])
print("Export complete!")
