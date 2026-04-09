import cv2
import os
import serial  # FIXED: Not 'pyserial'
import sys
import subprocess
import json
import numpy as np
import time
from ultralytics import YOLO
from stable_baselines3 import SAC

print("Loading all models!")
try:
    yolo_model = YOLO("best_cloth.engine", task='detect') 
    # NOTE: If this crashes due to numpy, change the name to the converted model!
    rl_model = SAC.load("sac_cayote_model_50000_steps") # FIXED: Name matched to the loop below
except Exception as e:
    print(f"Failed to load models! :( Error: {e}")
    sys.exit(5) 

print("Models loaded!")
print("Loading Arduino!")

arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=115200, timeout=0.1)
time.sleep(2) 

cap = cv2.VideoCapture(0)
camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

def read_arduino_distance():
    """Reads the JSON string from Arduino and extracts the 'front' distance."""
    front_dist = 10.0 
    try:
        if arduino.in_waiting > 0:
            line = arduino.readline().decode('utf-8').strip()
            if line.startswith("{") and line.endswith("}"):
                data = json.loads(line)
                front_dist = data.get("front", 10.0)
    except Exception as e:
        pass 
    return front_dist

def send_arduino_command(action):
    """Translates RL continuous output into Arduino character commands."""
    speed = float(action[0])
    steer = float(action[1])
    
    if speed > 0.3:
        speed_cmd = 'd' # Drive
    elif speed < -0.3:
        speed_cmd = 'v' # Reverse
    else:
        speed_cmd = 's' # Stop

    if steer > 0.3:
        steer_cmd = 'r' # Right
    elif steer < -0.3:
        steer_cmd = 'l' # Left
    else:
        steer_cmd = 'f' # Straight
        
    arduino.write(speed_cmd.encode())
    arduino.write(steer_cmd.encode())
    
    return speed_cmd, steer_cmd

print("System Active --> Q to quit!")
try:
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            continue
            
        results = yolo_model(frame, verbose=False)[0]
        found = 0.0
        offset = 0.0
        human_area_norm = 0.0
        
        for box in results.boxes: # FIXED: Was 'result_boxes'
            if int(box.cls[0]) == 0:
                found = 1.0
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w = x2 - x1
                h = y2 - y1
                
                human_area_norm = (w*h)/10000.0
                center_x = x1 + (w/2) 
                offset = (center_x - (camera_width/2))/(camera_width/2)
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                break
        
        dist = read_arduino_distance()
        obs = np.array([found, offset, human_area_norm, dist], dtype=np.float32)
        
        action, _ = rl_model.predict(obs, deterministic=True)
        
        if dist < 0.45 and float(action[0]) > 0:
            action[0] = 0.0 

        speed_c, steer_c = send_arduino_command(action)

        cv2.putText(frame, f"Dist: {dist:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        cv2.putText(frame, f"RL Action: Spd={action[0]:.2f}, Str={action[1]:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, f"Sent: '{speed_c}', '{steer_c}'", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        cv2.imshow("Cayote Brain HUD", frame)

        elapsed = time.time() - start_time
        sleep_time = max(0, 0.1 - elapsed) 
        time.sleep(sleep_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
               
except KeyboardInterrupt:
    print("Interrupted by user!")
    
finally:
    print("Shutting down... sending STOP to Arduino.")
    arduino.write('s'.encode())
    arduino.write('f'.encode())
    arduino.close()
    cap.release()
    cv2.destroyAllWindows()