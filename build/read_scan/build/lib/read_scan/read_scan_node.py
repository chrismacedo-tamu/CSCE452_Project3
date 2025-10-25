import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration


import numpy as np
import math

class ReadScanNode(Node):
    def __init__(self):
        super().__init__('read_scan_node')

        # Subscribing to /scan
        self.scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Publishing to /people_data
        self.people_data = self.create_publisher(PointStamped, '/people_data', 10)

        self.marker_pub = self.create_publisher(MarkerArray, '/people_markers', 10)


        # Using cluster detection to find moving people
        self.cluster_threshold = 1.5
        self.min_cluster_size = 8
        self.max_cluster_size = 30

    def scan_callback(self, msg):
        # Converting data to Cartesian coords 
        # OG data is in Polar coords
        angles = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
        ranges = np.array(msg.ranges)

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        points = np.vstack((x, y)).T

        # Finding clusters
        cluster_list = []
        cur_cluster = [points[0]]

        for i in range(1, len(points)):
            # Calculating distance from previous point
            dist = np.linalg.norm(points[i] - points[i-1])
            if dist < self.cluster_threshold:
                cur_cluster.append(points[i])
            else:
                if self.min_cluster_size <= len(cur_cluster) <= self.max_cluster_size:
                    cluster_list.append(np.array(cur_cluster))
                cur_cluster = [points[i]]

        # Checking last cluster
        if self.min_cluster_size <= len(cur_cluster) <= self.max_cluster_size:
            cluster_list.append(np.array(cur_cluster))
            
        for cluster in cluster_list:
            # Extracting center from each cluster
            center = np.mean(cluster, axis=0)

            # 
            point_msg = PointStamped()
            point_msg.header = Header()
            point_msg.header.stamp = self.get_clock().now().to_msg()
            point_msg.header.frame_id = "base_link"

            point_msg.point.x = float(center[0])
            point_msg.point.y = float(center[1])
            point_msg.point.z = 0.0

            self.people_data.publish(point_msg)

        marker_array = MarkerArray()

        # Code for visualizing in rviz2
        for i, cluster in enumerate(cluster_list):
            center = np.mean(cluster, axis=0)

            marker = Marker()
            marker.lifetime = Duration(sec=1)
            marker.frame_locked = False
            marker.header.frame_id = "laser"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "people"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(center[0])
            marker.pose.position.y = float(center[1])
            marker.pose.position.z = 0.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)




def main(args=None):
    rclpy.init(args=args)
    node = ReadScanNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()