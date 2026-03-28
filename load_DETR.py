import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from DETR import DETR_Deer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.7 

def run_inference():
    model = DETR_Deer(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load("DETR_Best.pth", map_location=DEVICE))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])

    cap = cv2.VideoCapture(0) 
    print("Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs_class, outputs_coord = model(img_tensor)


        probs = outputs_class.softmax(-1)[0, :, :-1] # Remove the 'no object' class
        keep = probs.max(-1).values > CONFIDENCE_THRESHOLD

        h, w, _ = frame.shape
        for p, (x_c, y_c, wb, hb) in zip(probs[keep], outputs_coord[0, keep]):
            xmin = int((x_c - 0.5 * wb) * w)
            ymin = int((y_c - 0.5 * hb) * h)
            xmax = int((x_c + 0.5 * wb) * w)
            ymax = int((y_c + 0.5 * hb) * h)

            # Draw the rectangle 
            cl = p.argmax().item()
            if cl == 0: # Class 0 is Human
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                label = f"Human: {p[cl]:.2f}"
                cv2.putText(frame, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('DETR Real-Time Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference()

