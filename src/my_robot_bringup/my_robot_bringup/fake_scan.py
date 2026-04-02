import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math

class FakeScan(Node):
    def __init__(self):
        super().__init__('fake_scan')
        self.publisher_ = self.create_publisher(LaserScan, '/scan', 10)
        self.timer = self.create_timer(0.1, self.publish_fake_scan)

    def publish_fake_scan(self):
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        # Bind it to the camera link from your URDF!
        msg.header.frame_id = 'camera'
        
        # 180 degree scan field of view
        msg.angle_min = -1.57 
        msg.angle_max = 1.57
        msg.angle_increment = 3.14 / 100.0
        msg.time_increment = 0.0
        msg.range_min = 0.1
        msg.range_max = 10.0
        
        # Create an array of 100 points
        ranges = []
        for i in range(100):
            # Pretend there is a wall 1.0 meter away in the center
            # and it angles away towards the edges
            angle = msg.angle_min + i * msg.angle_increment
            distance = 1.0 / math.cos(angle) if math.cos(angle) > 0.1 else 10.0
            
            # Add some fake noise
            if -0.5 < angle < 0.5:
                ranges.append(1.0) # Solid wall 1 meter ahead
            else:
                ranges.append(10.0) # Clear space on the sides
                
        msg.ranges = ranges
        self.publisher_.publish(msg)

def main():
    rclpy.init()
    node = FakeScan()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()