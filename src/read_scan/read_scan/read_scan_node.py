import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped, Pose, PoseArray
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
from collections import deque
import numpy as np
import math


class ReadScanNode(Node):
    def __init__(self):
        super().__init__('read_scan_node')

        # Subscriptions
        self.scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Publishers
        self.people_data = self.create_publisher(PointStamped, '/people_data', 10)
        self.debug_marker_pub = self.create_publisher(MarkerArray, '/debug_spheres', 10)
        self.centroid_pub = self.create_publisher(PoseArray, '/valid_clusters', 10)

        # Clustering params
        self.cluster_threshold = 0.4
        self.min_cluster_size = 5
        self.max_cluster_size = 35

        # Tracking store
        self.track_memory = {}
        self.next_id = 0
        self.frame_count = 0

        self.get_logger().info("ReadScanNode started - WITH FILTERING")

    def scan_callback(self, msg: LaserScan):
        self.frame_count += 1

        # Convert scan to cartesian
        n = len(msg.ranges)
        if n == 0:
            return
        angles = msg.angle_min + np.arange(n, dtype=float) * msg.angle_increment
        ranges = np.array(msg.ranges, dtype=float)

        # Filter ranges
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=0.0, neginf=0.0)
        ranges = np.clip(ranges, 0.05, 8.0)

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        points = np.vstack((x, y)).T

        if points.shape[0] == 0:
            return

        # Cluster consecutive points
        cluster_list = []
        cur_cluster = [points[0]]
        for i in range(1, len(points)):
            dist = np.linalg.norm(points[i] - points[i - 1])
            point_range = np.linalg.norm(points[i])
            adaptive_threshold = self.cluster_threshold * (1 + point_range * 0.1)
            if dist < adaptive_threshold:
                cur_cluster.append(points[i])
            else:
                if self.min_cluster_size <= len(cur_cluster) <= self.max_cluster_size:
                    cluster_list.append(np.array(cur_cluster))
                cur_cluster = [points[i]]
        if self.min_cluster_size <= len(cur_cluster) <= self.max_cluster_size:
            cluster_list.append(np.array(cur_cluster))

        # Filtering by shape
        filtered_clusters = []
        for cluster in cluster_list:
            if cluster.shape[0] >= 3:
                cov = np.cov(cluster.T)
                vals, _ = np.linalg.eig(cov)
                major = float(np.sqrt(max(vals.max(), 1e-9)))
                minor = float(np.sqrt(max(vals.min(), 1e-9)))
                if minor > 0 and (major / minor) < 5.0:
                    filtered_clusters.append(cluster)
            else:
                filtered_clusters.append(cluster)

        cur_centers = [np.mean(cluster, axis=0) for cluster in filtered_clusters]

        # Track clusters over time
        updated_tracks = {}
        matched_track_ids = set()

        # Match current cluster detections to existing tracks
        for center in cur_centers:
            best_track = None
            best_dist = float('inf')
            for tid, tdata in self.track_memory.items():
                dist = np.linalg.norm(center - tdata['last_pos'])
                if dist < best_dist and dist < 1.5:
                    best_dist = dist
                    best_track = tid

            if best_track is not None:
                # Update existing track
                old_data = self.track_memory[best_track]
                history = old_data['history'].copy()
                history.append(center)
                # Keep last 30 positions
                if len(history) > 30:
                    history.pop(0)
                
                updated_tracks[best_track] = {
                    'last_pos': center,
                    'initial_pos': old_data['initial_pos'],
                    'history': history,
                    'seen_count': old_data['seen_count'] + 1,
                    'missing': 0
                }
                matched_track_ids.add(best_track)
            else:
                # Create new track
                updated_tracks[self.next_id] = {
                    'last_pos': center,
                    'initial_pos': center,
                    'history': [center],
                    'seen_count': 1,
                    'missing': 0
                }
                self.next_id += 1

        # Keep unmatched tracks
        for tid, tdata in self.track_memory.items():
            if tid in matched_track_ids:
                continue
            missing = tdata.get('missing', 0) + 1
            if missing <= 20: 
                tcopy = dict(tdata)
                tcopy['missing'] = missing
                updated_tracks[tid] = tcopy

        self.track_memory = updated_tracks

        # Filter out static objects - CONTINUOUS TRACKING
        valid_centers = []
        for tid, tdata in self.track_memory.items():
            # Allow more missing frames to handle brief occlusions
            if tdata.get('missing', 0) > 8:
                continue
            
            history = tdata['history']
            seen = tdata['seen_count']
            
            # Strategy: Publish everything initially, then remove only persistently static objects
            
            if len(history) >= 12:
                # After 12+ frames, we have enough data to be confident
                positions = np.array(history)
                
                # Look at the FULL history to determine if truly static
                total_displacement = float(np.linalg.norm(positions[-1] - positions[0]))
                std_dev = float(np.mean(np.std(positions, axis=0)))
                
                # Also check recent movement (last 6 frames)
                recent_positions = positions[-6:]
                recent_displacement = float(np.linalg.norm(recent_positions[-1] - recent_positions[0]))
                
                # Calculate path length
                path_length = 0.0
                for i in range(1, len(positions)):
                    path_length += float(np.linalg.norm(positions[i] - positions[i-1]))
                
                # Mark as static ONLY if it's been sitting still the ENTIRE time
                is_persistently_static = (
                    total_displacement < 0.12 and      # barely moved from start
                    std_dev < 0.04 and                 # very stable
                    recent_displacement < 0.2 and     # not moving recently either
                    path_length < 0.20                 # almost no total path
                )
                
                if not is_persistently_static:
                    valid_centers.append(tdata['last_pos'])
            else:
                # For newer tracks, always publish
                #    ensures we capture full movement from start
                valid_centers.append(tdata['last_pos'])

        # Publish PoseArray to /valid_clusters
        pa = PoseArray()
        pa.header = msg.header
        for center in valid_centers:
            pose = Pose()
            pose.position.x = float(center[0])
            pose.position.y = float(center[1])
            pose.position.z = 0.0
            pose.orientation.w = 1.0
            pa.poses.append(pose)
        self.centroid_pub.publish(pa)
        
        # Log
        if self.frame_count % 30 == 0:
            num_static = len(self.track_memory) - len(valid_centers)
            self.get_logger().info(f"Tracking: {len(self.track_memory)}, Publishing: {len(valid_centers)}, Filtered: {num_static}")


def main(args=None):
    rclpy.init(args=args)
    node = ReadScanNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()