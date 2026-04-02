import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry
import math

class IsaacVslamSimulator(Node):
    def __init__(self):
        super().__init__('vslam_simulator_test')
        
        # Isaac ROS VSLAM usually uses Reliable QoS
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )

        # Publishers
        self.imu_pub = self.create_publisher(Imu, '/imu/data', reliable_qos)
        self.vslam_pub = self.create_publisher(Odometry, '/visual_slam/tracking/odometry', reliable_qos)
        self.gps_pub = self.create_publisher(NavSatFix, '/gps/fix', 10)

        # Movement Variables
        self.timer = self.create_timer(0.05, self.update_robot_physics) # 20Hz
        self.angle = 0.0
        self.radius = 2.0 # 2 meter circle
        
        self.get_logger().info("VSLAM Simulator Active. Robot is now walking in a circle...")

    def update_robot_physics(self):
        now = self.get_clock().now().to_msg()
        self.angle += 0.04 

        # CORRECTED MATH: 
        # We subtract the radius from X so the first point (where angle=0) is (0,0)
        x = (self.radius * math.cos(self.angle)) - self.radius
        y = self.radius * math.sin(self.angle)
        
        # Orientation: facing the direction of travel
        yaw = self.angle + (math.pi / 2.0)

        vslam = Odometry()
        vslam.header.stamp = now
        vslam.header.frame_id = 'odom'
        vslam.child_frame_id = 'base_link'
        
        vslam.pose.pose.position.x = x
        vslam.pose.pose.position.y = y
        
        # Quaternion for Yaw
        vslam.pose.pose.orientation.z = math.sin(yaw / 2.0)
        vslam.pose.pose.orientation.w = math.cos(yaw / 2.0)
        
        self.vslam_pub.publish(vslam)
        
        # Also publish IMU so the EKF doesn't get confused
        imu = Imu()
        imu.header.stamp = now
        imu.header.frame_id = 'base_link'
        imu.orientation = vslam.pose.pose.orientation
        self.imu_pub.publish(imu)

        # 4. Keep GPS Lock constant
        gps = NavSatFix()
        gps.header.stamp = now
        gps.header.frame_id = 'gps'
        gps.latitude = 37.380
        gps.longitude = -121.968
        self.gps_pub.publish(gps)

def main():
    rclpy.init()
    node = IsaacVslamSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()