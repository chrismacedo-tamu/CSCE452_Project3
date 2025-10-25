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
        self.track_memory = {}
        self.next_id = 0
        self.cluster_threshold = 1.0
        self.min_cluster_size = 5
        self.max_cluster_size = 35
        
        # Detection parameters
        self.static_position_threshold = 0.08
        self.static_check_frames = 10
        self.smoothing_alpha = 0.7
        
        # Static removal parameters
        self.check_interval = 20  # Check for static tracks every N frames
        self.frame_count = 0
        
        self.last_time = None

    def scan_callback(self, msg):
        self.frame_count += 1
        
        # Converting data to Cartesian coords 
        angles = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
        ranges = np.array(msg.ranges)

        # Filtering LIDAR data
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=0.0, neginf=0.0)
        ranges = np.clip(ranges, 0.05, 8.0)

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        points = np.vstack((x, y)).T

        if points.shape[0] == 0:
            return

        # Finding clusters - use adaptive threshold
        cluster_list = []
        cur_cluster = [points[0]]

        for i in range(1, len(points)):
            dist = np.linalg.norm(points[i] - points[i-1])
            
            # Adaptive clustering threshold based on distance from sensor
            point_range = np.linalg.norm(points[i])
            adaptive_threshold = self.cluster_threshold * (1 + point_range * 0.1)
            
            if dist < adaptive_threshold:
                cur_cluster.append(points[i])
            else:
                if self.min_cluster_size <= len(cur_cluster) <= self.max_cluster_size:
                    cluster_list.append(np.array(cur_cluster))
                cur_cluster = [points[i]]

        # Checking last cluster
        if self.min_cluster_size <= len(cur_cluster) <= self.max_cluster_size:
            cluster_list.append(np.array(cur_cluster))
            
        # Shape filter
        filtered_clusters = []
        for cluster in cluster_list:
            if cluster.shape[0] >= 3:
                cov = np.cov(cluster.T)
                vals, _ = np.linalg.eig(cov)
                major = np.sqrt(vals.max())
                minor = np.sqrt(vals.min())

                if (minor > 0):
                    shape_ratio = major / minor
                    if shape_ratio < 5:
                        filtered_clusters.append(cluster)
            else:
                filtered_clusters.append(cluster)

        cur_centers = [np.mean(cluster, axis=0) for cluster in filtered_clusters]
        cur_time = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9

        # Initialize tracking
        if not hasattr(self, 'track_memory'):
            self.track_memory = {}
            self.next_id = 0

        # Tracking parameters
        max_missing = 20
        min_seen_for_publish = 2
        min_seen_for_static_check = 10
        
        # Calculate time delta
        time_delta = 0.1
        if self.last_time is not None:
            time_delta = max(cur_time - self.last_time, 0.01)
        self.last_time = cur_time
        
        updated_tracks = {}
        matched_track_ids = set()
        
        # Match clusters to tracks
        for center in cur_centers:
            best_track = None
            best_dist = float('inf')

            for tid, tdata in self.track_memory.items():
                # Use smoothed position for matching
                match_pos = tdata.get('smoothed_pos', tdata['pos'])
                dist = np.linalg.norm(center - match_pos)
                
                # Adaptive matching distance
                missing_frames = tdata.get('missing', 0)
                if tdata.get('confirmed_moving', False):
                    max_dist = 1.8 + (missing_frames * 0.25)
                elif tdata['seen_count'] >= min_seen_for_publish:
                    max_dist = 1.2 + (missing_frames * 0.2)
                else:
                    max_dist = 0.7
                
                if dist < best_dist and dist < max_dist:
                    best_dist = dist
                    best_track = tid
            
            if best_track is not None:
                old_data = self.track_memory[best_track]
                
                # Apply exponential smoothing
                if 'smoothed_pos' in old_data:
                    smoothed_pos = (self.smoothing_alpha * center + 
                                  (1 - self.smoothing_alpha) * old_data['smoothed_pos'])
                else:
                    smoothed_pos = center
                
                # Track raw positions for static detection
                raw_position_history = old_data.get('raw_position_history', [])
                raw_position_history.append(center)
                raw_position_history = raw_position_history[-self.static_check_frames:]
                
                # Get initial position
                initial_pos = old_data.get('initial_pos', center)
                distance_from_start = np.linalg.norm(center - initial_pos)
                
                # Calculate position variance
                if len(raw_position_history) >= self.static_check_frames:
                    positions = np.array(raw_position_history)
                    position_std = np.std(positions, axis=0)
                    position_variance = np.mean(position_std)
                    
                    mean_pos = np.mean(positions, axis=0)
                    max_deviation = np.max(np.linalg.norm(positions - mean_pos, axis=1))
                    
                    if len(raw_position_history) >= 5:
                        mid = len(raw_position_history) // 2
                        first_half_mean = np.mean(positions[:mid], axis=0)
                        second_half_mean = np.mean(positions[mid:], axis=0)
                        directional_movement = np.linalg.norm(second_half_mean - first_half_mean)
                    else:
                        directional_movement = 0.0
                else:
                    position_variance = 0.0
                    max_deviation = 0.0
                    directional_movement = 0.0
                
                instant_movement = np.linalg.norm(center - old_data['pos'])
                
                updated_tracks[best_track] = {
                    'pos': center,
                    'smoothed_pos': smoothed_pos,
                    'initial_pos': initial_pos,
                    'distance_from_start': distance_from_start,
                    'raw_position_history': raw_position_history,
                    'position_variance': position_variance,
                    'max_deviation': max_deviation,
                    'directional_movement': directional_movement,
                    'instant_movement': instant_movement,
                    'seen_count': old_data['seen_count'] + 1,
                    'time': cur_time,
                    'missing': 0,
                    'confirmed_moving': old_data.get('confirmed_moving', False)
                }
                matched_track_ids.add(best_track)
            else:
                # New track
                updated_tracks[self.next_id] = {
                    'pos': center,
                    'smoothed_pos': center,
                    'initial_pos': center,
                    'distance_from_start': 0.0,
                    'raw_position_history': [center],
                    'position_variance': 0.0,
                    'max_deviation': 0.0,
                    'directional_movement': 0.0,
                    'instant_movement': 0.0,
                    'seen_count': 1,
                    'time': cur_time,
                    'missing': 0,
                    'confirmed_moving': False
                }
                self.next_id += 1

        # Keep unmatched tracks
        for tid, tdata in self.track_memory.items():
            if tid in matched_track_ids:
                continue
            missing = tdata.get('missing', 0) + 1
            
            # Keep longer if confirmed moving
            max_keep = max_missing if tdata.get('confirmed_moving', False) else max_missing // 2
            
            if missing <= max_keep:
                updated_tracks[tid] = {
                    'pos': tdata['pos'],
                    'smoothed_pos': tdata.get('smoothed_pos', tdata['pos']),
                    'initial_pos': tdata.get('initial_pos', tdata['pos']),
                    'distance_from_start': tdata.get('distance_from_start', 0.0),
                    'raw_position_history': tdata.get('raw_position_history', []),
                    'position_variance': tdata.get('position_variance', 0.0),
                    'max_deviation': tdata.get('max_deviation', 0.0),
                    'directional_movement': tdata.get('directional_movement', 0.0),
                    'instant_movement': 0.0,
                    'seen_count': tdata['seen_count'],
                    'time': tdata['time'],
                    'missing': missing,
                    'confirmed_moving': tdata.get('confirmed_moving', False)
                }

        self.track_memory = updated_tracks
        
        # Periodic cleanup: Remove static tracks from memory
        if self.frame_count % self.check_interval == 0:
            tracks_to_remove = []
            for tid, tdata in self.track_memory.items():
                # Only check tracks that have enough history
                if tdata['seen_count'] >= min_seen_for_static_check:
                    variance = tdata.get('position_variance', 0.0)
                    max_dev = tdata.get('max_deviation', 0.0)
                    dir_movement = tdata.get('directional_movement', 0.0)
                    dist_from_start = tdata.get('distance_from_start', 0.0)
                    
                    # Identify as static if all criteria met
                    is_static = (variance < self.static_position_threshold and 
                               max_dev < 0.15 and 
                               dir_movement < 0.12 and
                               dist_from_start < 0.25)
                    
                    if is_static and not tdata.get('confirmed_moving', False):
                        tracks_to_remove.append(tid)
            
            # Remove static tracks
            for tid in tracks_to_remove:
                if tid in self.track_memory:
                    del self.track_memory[tid]

        # Select valid tracks
        valid_centers = []
        for tid, tdata in self.track_memory.items():
            # Must be recently visible
            max_missing_for_publish = 6 if tdata.get('confirmed_moving', False) else 2
            if tdata.get('missing', 0) > max_missing_for_publish:
                continue
            
            # Need minimum observations
            if tdata['seen_count'] < min_seen_for_publish:
                continue
            
            is_moving = False
            
            # Check if has enough history for static determination
            if tdata['seen_count'] >= min_seen_for_static_check:
                variance = tdata.get('position_variance', 0.0)
                max_dev = tdata.get('max_deviation', 0.0)
                dir_movement = tdata.get('directional_movement', 0.0)
                
                # Static if: low variance AND low deviation AND no directional movement
                is_static = (variance < self.static_position_threshold and 
                           max_dev < 0.15 and 
                           dir_movement < 0.12)
                
                if not is_static:
                    is_moving = True
                    tdata['confirmed_moving'] = True
            else:
                # Not enough data yet - be optimistic
                if tdata.get('instant_movement', 0) > 0.08:
                    is_moving = True
            
            # Once confirmed moving, keep tracking
            if tdata.get('confirmed_moving', False):
                is_moving = True
            
            if is_moving:
                valid_centers.append(tdata['smoothed_pos'])

        # Publishing data
        for center in valid_centers:
            point_msg = PointStamped()
            point_msg.header.stamp = self.get_clock().now().to_msg()
            point_msg.header.frame_id = "base_link"
            point_msg.point.x = float(center[0])
            point_msg.point.y = float(center[1])
            point_msg.point.z = 0.0
            self.people_data.publish(point_msg)

        # Visualize in RViz
        marker_array = MarkerArray()
        for i, center in enumerate(valid_centers):
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