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

        # Setting up subscriptions
        self.scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.people_data = self.create_publisher(PointStamped, '/people_data', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/people_markers', 10)

        # Tracking clusters over time
        self.track_memory = {}
        self.next_id = 0

        # Cluster detection parameters
        self.cluster_thresh = 0.8
        self.min_cluster_size = 5
        self.max_cluster_size = 35

        # Moving cluster parameters
        self.static_position_threshold = 0.08
        self.static_frames = 12
        self.smoothing_alpha = 0.6

        self.check_interval = 25
        self.frame_count = 0

        self.last_time = None

    def scan_callback(self, msg):
        self.frame_count += 1
        
        # Converting to Cartesian coords from polar coords
        angles = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
        ranges = np.array(msg.ranges)

        # Filtering lidar data
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=0.0, neginf=0.0)
        ranges = np.clip(ranges, 0.05, 8.0)

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        points = np.vstack((x,y)).T

        # Finding clusters based on point distance and shapes
        cluster_list = []
        cur_cluster = [points[0]]

        for i in range(1, len(points)):
            dist = np.linalg.norm(points[i] - points[i-1])

            # Adaptive clustering
            # Loose calculations for further objects
            point_range = np.linalg.norm(points[i])
            adaptive_thresh = self.cluster_thresh * (1 + point_range * 0.08)

            if dist < adaptive_thresh:
                cur_cluster.append(points[i])
            else:
                # Using # of points in 'cluster' to accept clusters
                if self.min_cluster_size <= len(cur_cluster) <= self.max_cluster_size:
                    arr = np.array(cur_cluster)
                    cluster_list.append(arr)
                cur_cluster = [points[i]]

        # Checking last cluster
        if self.min_cluster_size <= len(cur_cluster) <= self.max_cluster_size:
            arr = np.array(cur_cluster)
            cluster_list.append(arr)
        
        # filtering clusters based off of shape
        # Straight lines = walls, essentially
        filtered_clusters = []
        for cluster in cluster_list:
            if cluster.shape[0] >= 3:
                # Wall-like object
                cov = np.cov(cluster.T)
                vals, _ = np.linalg.eig(cov)
                major = np.sqrt(vals.max())
                minor = np.sqrt(vals.min())

                if (minor>0):
                    shape_ratio = major/minor
                    if shape_ratio < 4:
                        filtered_clusters.append(cluster)
            else:
                filtered_clusters.append(cluster)

        cur_centers = [np.mean(cluster, axis=0) for cluster in filtered_clusters]
        cur_time = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9

        # using cluster history to determine moving clusters
        max_missing = 20
        min_publish_seen = 3
        min_static_seen = 12

        self.last_time = cur_time

        # Matching clusters to existing, moving tracks
        updated_tracks = {}
        matched_ids = set()
        confirmed_moving_tracks = {}
        non_moving_tracks = {}
        for tid, t in self.track_memory.items():
            if t.get('confirmed_moving', False):
                confirmed_moving_tracks[tid] = t
            else:
                non_moving_tracks[tid] = t
        
        # Match confimred moving tracks with flexible threshs
        for center in cur_centers:
            best_track = None
            best_dist = float('inf')

            for tid, tdata in confirmed_moving_tracks.items():
                if tid in matched_ids:
                    continue
                match_pos = tdata.get('smoothed_pos', tdata['pos'])
                dist = np.linalg.norm(center-match_pos)

                missing_frames = tdata.get('missing', 0)
                max_dist = 2.0 + missing_frames * 0.3

                if dist < best_dist and dist < max_dist:
                    best_dist = dist
                    best_track = tid
            if best_track is not None:
                old_data = self.track_memory[best_track]

                # Less aggressive smoothing
                if 'smoothed_pos' in old_data:
                    smoothed_pos = (self.smoothing_alpha * center + (1-self.smoothing_alpha) * old_data['smoothed_pos'])
                else:
                    smoothed_pos = center
                
                raw_pos_history = old_data.get('raw_pos_history', [])
                raw_pos_history.append(center)
                raw_pos_history = raw_pos_history[-self.static_frames:]

                # Extracting initial pos
                initial_pos = old_data.get('initial_pos', center)
                dist_from_start = np.linalg.norm(center-initial_pos)

                # Position variance calculation
                if len(raw_pos_history) >= self.static_frames:
                    positions = np.array(raw_pos_history)
                    position_std = np.std(positions, axis=0)
                    position_variance = np.mean(position_std)
                    
                    mean_pos = np.mean(positions, axis=0)
                    max_deviation = np.max(np.linalg.norm(positions - mean_pos, axis=1))
                    
                    if len(raw_pos_history) >= 5:
                        mid = len(raw_pos_history) // 2
                        first_mean = np.mean(positions[:mid], axis=0)
                        second_mean = np.mean(positions[mid:], axis=0)
                        directional_movement = np.linalg.norm(second_mean - first_mean)
                    else:
                        directional_movement = 0.0
                else:
                    position_variance = 0.0
                    max_deviation = 0.0
                    directional_movement = 0.0
                
                instant_movement = np.linalg.norm(center - old_data['pos'])
                
                # Updating tracks
                updated_tracks[best_track] = {
                    'pos': center,
                    'smoothed_pos': smoothed_pos,
                    'initial_pos': initial_pos,
                    'dist_from_start': dist_from_start,
                    'raw_pos_history': raw_pos_history,
                    'position_variance': position_variance,
                    'max_deviation': max_deviation,
                    'directional_movement': directional_movement,
                    'instant_movement': instant_movement,
                    'seen_count': old_data['seen_count'] + 1,
                    'time': cur_time,
                    'missing': 0,
                    'confirmed_moving': True  # Keep confirmed status
                }
                matched_ids.add(best_track)
        
        # Then match other tracks with stricter thresholds
        for center in cur_centers:
            skip = False
            for tid in matched_ids:
                if center.tobytes() == updated_tracks[tid]['pos'].tobytes():
                    skip = True
                    break
            if skip:
                continue
                
            best_track = None
            best_dist = float('inf')

            for tid, tdata in non_moving_tracks.items():
                if tid in matched_ids:
                    continue
                    
                match_pos = tdata.get('smoothed_pos', tdata['pos'])
                dist = np.linalg.norm(center - match_pos)
                
                missing_frames = tdata.get('missing', 0)
                if tdata['seen_count'] >= min_publish_seen:
                    max_dist = 1.0 + (missing_frames * 0.15)
                else:
                    max_dist = 0.6
                
                if dist < best_dist and dist < max_dist:
                    best_dist = dist
                    best_track = tid
            
            if best_track is not None:
                old_data = self.track_memory[best_track]
                
                if 'smoothed_pos' in old_data:
                    smoothed_pos = (self.smoothing_alpha * center + 
                                  (1 - self.smoothing_alpha) * old_data['smoothed_pos'])
                else:
                    smoothed_pos = center
                
                raw_pos_history = old_data.get('raw_pos_history', [])
                raw_pos_history.append(center)
                raw_pos_history = raw_pos_history[-self.static_frames:]
                
                initial_pos = old_data.get('initial_pos', center)
                dist_from_start = np.linalg.norm(center - initial_pos)
                
                if len(raw_pos_history) >= self.static_frames:
                    positions = np.array(raw_pos_history)
                    position_std = np.std(positions, axis=0)
                    position_variance = np.mean(position_std)
                    
                    mean_pos = np.mean(positions, axis=0)
                    max_deviation = np.max(np.linalg.norm(positions - mean_pos, axis=1))
                    
                    if len(raw_pos_history) >= 5:
                        mid = len(raw_pos_history) // 2
                        first_mean = np.mean(positions[:mid], axis=0)
                        second_mean = np.mean(positions[mid:], axis=0)
                        directional_movement = np.linalg.norm(second_mean - first_mean)
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
                    'dist_from_start': dist_from_start,
                    'raw_pos_history': raw_pos_history,
                    'position_variance': position_variance,
                    'max_deviation': max_deviation,
                    'directional_movement': directional_movement,
                    'instant_movement': instant_movement,
                    'seen_count': old_data['seen_count'] + 1,
                    'time': cur_time,
                    'missing': 0,
                    'confirmed_moving': old_data.get('confirmed_moving', False)
                }
                matched_ids.add(best_track)
            else:
                # Creating new track if far enough from other tracks
                min_dist_to_existing = float('inf')
                for tid in updated_tracks:
                    dist = np.linalg.norm(center - updated_tracks[tid]['pos'])
                    min_dist_to_existing = min(min_dist_to_existing, dist)
                
                if min_dist_to_existing > 0.3:
                    updated_tracks[self.next_id] = {
                        'pos': center,
                        'smoothed_pos': center,
                        'initial_pos': center,
                        'dist_from_start': 0.0,
                        'raw_pos_history': [center],
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
            if tid in matched_ids:
                continue
            missing = tdata.get('missing', 0) + 1
            
            # Keeping confirmed tracks for longer
            if tdata.get('confirmed_moving', False):
                max_keep = max_missing
            else:
                max_keep = max_missing // 2
            
            # Updating tracks
            if missing <= max_keep:
                updated_tracks[tid] = {
                    'pos': tdata['pos'],
                    'smoothed_pos': tdata.get('smoothed_pos', tdata['pos']),
                    'initial_pos': tdata.get('initial_pos', tdata['pos']),
                    'dist_from_start': tdata.get('dist_from_start', 0.0),
                    'raw_pos_history': tdata.get('raw_pos_history', []),
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
        
        # Cleaning up static tracks
        if self.frame_count % self.check_interval == 0:
            tracks_to_remove = []
            for tid, tdata in self.track_memory.items():
                # Never remove confirmed moving tracks
                if tdata.get('confirmed_moving', False):
                    continue
                
                # Check tracks with enough history
                if tdata['seen_count'] >= min_static_seen:
                    variance = tdata.get('position_variance', 0.0)
                    max_dev = tdata.get('max_deviation', 0.0)
                    dir_movement = tdata.get('directional_movement', 0.0)
                    dist_from_start = tdata.get('dist_from_start', 0.0)
                    
                    # Stricter static detection
                    is_static = (variance < self.static_position_threshold and 
                               max_dev < 0.12 and 
                               dir_movement < 0.10 and
                               dist_from_start < 0.25)
                    
                    if is_static:
                        tracks_to_remove.append(tid)
                
                # Removing barely moving tracks
                if tdata['seen_count'] >= 20:
                    if tdata.get('dist_from_start', 0.0) < 0.4:
                        tracks_to_remove.append(tid)
            
            for tid in tracks_to_remove:
                if tid in self.track_memory:
                    del self.track_memory[tid]

        # Select valid tracks for publishing
        valid_centers = []
        for tid, tdata in self.track_memory.items():
            # Must be recently visible
            max_missing_for_publish = 8 if tdata.get('confirmed_moving', False) else 3
            if tdata.get('missing', 0) > max_missing_for_publish:
                continue
            if tdata['seen_count'] < min_publish_seen:
                continue
            
            is_moving = False
            
            # Check if enough history to mark as static
            if tdata['seen_count'] >= min_static_seen:
                variance = tdata.get('position_variance', 0.0)
                max_dev = tdata.get('max_deviation', 0.0)
                dir_movement = tdata.get('directional_movement', 0.0)
                dist_from_start = tdata.get('dist_from_start', 0.0)
                
                if dist_from_start > 0.35 or max_dev > 0.2 or dir_movement > 0.15:
                    is_moving = True
                    tdata['confirmed_moving'] = True
                # not static = moving
                elif not (variance < self.static_position_threshold and 
                         max_dev < 0.12 and 
                         dir_movement < 0.10):
                    is_moving = True
                    tdata['confirmed_moving'] = True
            else:
                # Not enough data
                if tdata.get('instant_movement', 0) > 0.06:
                    is_moving = True
                if tdata.get('confirmed_moving', False):
                    is_moving = True
            
            # Once confirmed moving, always publish
            if tdata.get('confirmed_moving', False):
                is_moving = True
            
            if is_moving:
                valid_centers.append(tdata.get('smoothed_pos', tdata['pos']))

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
        
        # Delete old markers first
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        for i, center in enumerate(valid_centers):
            marker = Marker()
            marker.lifetime = Duration(sec=0, nanosec=300000000)
            marker.frame_locked = False
            marker.header.frame_id = "laser"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "people_markers"
            marker.id = i + 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(center[0])
            marker.pose.position.y = float(center[1])
            marker.pose.position.z = 0.0
            marker.scale.x = 0.25
            marker.scale.y = 0.25
            marker.scale.z = 0.25
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = ReadScanNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()