#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Float32 
from stable_baselines3 import SAC

class CayoteInferenceNode:
    def __init__(self):
        rospy.init_node('cayote_rl_brain', anonymous=True)
        
        rospy.loginfo("Loading SAC Model into Jetson Memory...")
        # Load the trained model from the zip file
        self.model = SAC.load("sac_cayote_final")
        rospy.loginfo("Model Loaded Successfully!")
        
        # Internal state variables (default to safe values)
        self.found = 0.0
        self.offset = 0.0
        self.human_area_norm = 0.0
        self.dist = 10.0
        
        # --- PUBLISHER ---
        # This sends the speed/steering to your Arduino motor controller
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # --- SUBSCRIBERS ---
        # TODO: Change these topic names to match your actual ROS setup!
        rospy.Subscriber('/yolo/human_data', Float32MultiArray, self.yolo_callback)
        rospy.Subscriber('/arduino/ultrasonic_front', Float32, self.sonar_callback)
        
        # Run the loop at 10 Hz (similar to the Webots timestep)
        self.rate = rospy.Rate(10) 
        
    def yolo_callback(self, msg):
        # Assuming your YOLO node publishes: [found(0 or 1), offset, raw_area]
        self.found = msg.data[0]
        self.offset = msg.data[1]
        
        # CRITICAL: We must normalize the area exactly like we did in Webots!
        raw_area = msg.data[2]
        self.human_area_norm = raw_area / 10000.0 
        
    def sonar_callback(self, msg):
        # Assuming your Arduino publishes distance in meters
        self.dist = msg.data
        
    def run(self):
        rospy.loginfo("Cayote Brain is now driving!")
        
        while not rospy.is_shutdown():
            # 1. Build the Observation Array EXACTLY as it was in Webots
            obs = np.array([self.found, self.offset, self.human_area_norm, self.dist], dtype=np.float32)
            
            # 2. Ask the Neural Network what to do
            # deterministic=True tells the AI to stop taking random training guesses and use its best knowledge
            action, _states = self.model.predict(obs, deterministic=True)
            
            # 3. Scale the outputs back to real-world limits (matches Webots step() function)
            speed = float(action[0]) * 10.0
            steer = float(action[1]) * 0.4
            
            # 4. HARDCODED REAL-WORLD SAFETY OVERRIDE
            # Never trust AI completely on real hardware. If a wall is too close, force the brakes.
            if self.dist < 0.45 and speed > 0:
                speed = 0.0
                rospy.logwarn("Safety Override: Obstacle too close. Braking!")
                
            # 5. Convert to standard ROS motor commands
            twist = Twist()
            twist.linear.x = speed  # Forward/Backward
            twist.angular.z = steer # Steering (Yaw)
            
            # 6. Send to the Arduino
            self.cmd_vel_pub.publish(twist)
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = CayoteInferenceNode()
        node.run()
    except rospy.ROSInterruptException:
        # If the script is killed, send a final stop command to the motors
        stop_twist = Twist()
        rospy.Publisher('/cmd_vel', Twist, queue_size=10).publish(stop_twist)
        print("Cayote Brain Shutting Down safely.")
