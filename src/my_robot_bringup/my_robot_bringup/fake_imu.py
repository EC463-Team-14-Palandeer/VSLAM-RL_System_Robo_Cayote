import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import math
import time

class FakeImu(Node):
    def __init__(self):
        super().__init__('fake_imu')
        self.publisher_ = self.create_publisher(Imu, '/imu/data', 10)
        self.timer = self.create_timer(0.1, self.publish_fake_data)
        self.start_time = time.time()

    def publish_fake_data(self):
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu'
        
        # Calculate a fake pitch and yaw
        t = time.time() - self.start_time
        pitch = math.sin(t) * 0.5 
        yaw = math.cos(t) * 0.5 
        
        # Convert Euler angles to Quaternion 
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)

        msg.orientation.w = 1.0 * cp * cy + 0.0 * sp * sy
        msg.orientation.x = 0.0 * cp * cy - 1.0 * sp * sy
        msg.orientation.y = 1.0 * sp * cy + 0.0 * cp * sy
        msg.orientation.z = 1.0 * cp * sy - 0.0 * sp * cy
        
        # Add Covariance (Trust me, I'm a highly accurate fake sensor)
        msg.orientation_covariance[0] = 0.01  # X variance
        msg.orientation_covariance[4] = 0.01  # Y variance
        msg.orientation_covariance[8] = 0.01  # Z variance
        
        self.publisher_.publish(msg)

def main():
    rclpy.init()
    node = FakeImu()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()