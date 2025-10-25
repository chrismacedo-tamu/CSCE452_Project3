import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, PoseArray   # <-- add Pose/PoseArray
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
import numpy as np
import math

class ReadScanNode(Node):
    def __init__(self):
        super().__init__('read_scan_node')

        self.scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Optional: keep if you need it
        # self.people_data = self.create_publisher(PointStamped, '/people_data', 10)

        # DEBUG SPHERES ONLY (move off /person_markers)
        self.debug_marker_pub = self.create_publisher(MarkerArray, '/debug_spheres', 10)

        # <-- NEW: centroids for tracker
        self.centroid_pub = self.create_publisher(PoseArray, '/valid_clusters', 10)

        self.cluster_threshold = 1.5
        self.min_cluster_size = 8
        self.max_cluster_size = 30

    def scan_callback(self, msg: LaserScan):
        # Build angles and sanitize ranges
        n = len(msg.ranges)
        angles = msg.angle_min + np.arange(n) * msg.angle_increment
        ranges = np.asarray(msg.ranges, dtype=float)
        mask = np.isfinite(ranges)
        angles = angles[mask]
        ranges = ranges[mask]

        if ranges.size == 0:
            return

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        points = np.vstack((x, y)).T

        # Simple 1D adjacency clustering along the scan order
        cluster_list = []
        cur_cluster = [points[0]]
        for i in range(1, len(points)):
            dist = np.linalg.norm(points[i] - points[i - 1])
            if dist < self.cluster_threshold:
                cur_cluster.append(points[i])
            else:
                if self.min_cluster_size <= len(cur_cluster) <= self.max_cluster_size:
                    cluster_list.append(np.array(cur_cluster))
                cur_cluster = [points[i]]
        if self.min_cluster_size <= len(cur_cluster) <= self.max_cluster_size:
            cluster_list.append(np.array(cur_cluster))

        # ---- Publish centroids as PoseArray (for tracker) ----
        pa = PoseArray()
        pa.header = msg.header                 # same frame as LaserScan (e.g., "laser")
        pa.poses = []
        for cluster in cluster_list:
            cx, cy = np.mean(cluster, axis=0)
            pose = Pose()
            pose.position.x = float(cx)
            pose.position.y = float(cy)
            pose.position.z = 0.0
            pose.orientation.w = 1.0           # identity
            pa.poses.append(pose)
        self.centroid_pub.publish(pa)

        # ---- OPTIONAL: publish debug spheres (NOT /person_markers) ----
        marker_array = MarkerArray()
        for i, cluster in enumerate(cluster_list):
            cx, cy = np.mean(cluster, axis=0)
            m = Marker()
            m.lifetime = Duration(sec=1)
            m.header = msg.header
            m.ns = "people_debug"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(cx)
            m.pose.position.y = float(cy)
            m.pose.position.z = 0.0
            m.scale.x = m.scale.y = m.scale.z = 0.2
            m.color.a = 1.0; m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0
            marker_array.markers.append(m)
        self.debug_marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = ReadScanNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
