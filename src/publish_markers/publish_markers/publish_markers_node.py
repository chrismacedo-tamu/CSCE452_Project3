import math
import rclpy
from rclpy.node import Node
from scipy.spatial import KDTree
from collections import deque
from geometry_msgs.msg import PoseArray, Point
from visualization_msgs.msg import Marker, MarkerArray

GATE = 1.8 # max association distance (m) - increased to handle occlusions better
MAX_HISTORY = 200 # max points kept per person
HIDE_AFTER = 10 # frames before we stop extending a track - increased to maintain ID longer
SMOOTH_ALPHA = 0.1 # used for moving average

class PersonMarkers(Node):
    def __init__(self):
        super().__init__('publish_markers_node')

        #params so you can change via launch if needed
        self.declare_parameter('input_topic', '/valid_clusters')
        self.declare_parameter('frame_id', 'laser')

        in_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        self.sub = self.create_subscription(PoseArray, in_topic, self.on_clusters, 10)
        self.pub = self.create_publisher(MarkerArray, '/person_markers', 10)

        # tracks: id -> {'pos': (x,y), 'history': [Point,...], 'missed':int}
        self.tracks = {}
        self.next_id = 1
        
        # Define distinct colors for different people
        self.colors = [
            (0.2, 0.8, 0.2),   # Green
            (0.8, 0.2, 0.2),   # Red
            (0.2, 0.2, 0.8),   # Blue
            (0.8, 0.8, 0.2),   # Yellow
            (0.8, 0.2, 0.8),   # Magenta
            (0.2, 0.8, 0.8),   # Cyan
            (0.8, 0.5, 0.2),   # Orange
            (0.5, 0.2, 0.8),   # Purple
            (0.2, 0.8, 0.5),   # Teal
            (0.8, 0.8, 0.8),   # White
        ]
        
        self.get_logger().info(f"Publishing trails to /person_markers (LINE_STRIP). "
                               f"Subscribing to {in_topic} (PoseArray).")
        
            

    def on_clusters(self, msg: PoseArray):
        detections = [(p.position.x, p.position.y) for p in msg.poses]
        if not detections:
            return

        # Build KD tree with track positions
        if self.tracks:
            track_positions = [self.tracks[tid]['pos'] for tid in self.tracks]
            tids = list(self.tracks.keys())
            tree = KDTree(track_positions)

            matched_tids = set()
            assignments = []

            for (x, y) in detections:
                d, idx = tree.query((x, y))
                if d <= GATE:
                    tid = tids[idx]
                    if tid not in matched_tids:
                        matched_tids.add(tid)
                        assignments.append((tid, (x, y)))
                else:
                    # start a new track
                    tid = self.next_id
                    self.next_id += 1
                    self.tracks[tid] = {'pos': (x, y), 'history': [], 'missed': 0}
                    assignments.append((tid, (x, y)))
        else:
            assignments = []
            for (x, y) in detections:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {'pos': (x, y), 'history': [], 'missed': 0}
                assignments.append((tid, (x, y)))


        # update matched tracks
        for tid, (x,y) in assignments:
            tr = self.tracks[tid]
            old_x, old_y = tr['pos']
            new_x = SMOOTH_ALPHA * x + (1 - SMOOTH_ALPHA) * old_x
            new_y = SMOOTH_ALPHA * y + (1 - SMOOTH_ALPHA) * old_y
            tr['pos'] = (new_x, new_y)
            tr['missed'] = 0
        
        # increment missed for tracks that werent matched
        for tid in set(self.tracks.keys()) - {t for t, _ in assignments}:
            self.tracks[tid]['missed'] += 1
        
        # append cur pos to history only for visible tracks
        stamp = msg.header.stamp
        fr = msg.header.frame_id or self.frame_id

        for tid, tr in self.tracks.items():
            if tr['missed'] < HIDE_AFTER:
                p = Point()
                p.x, p.y, p.z = tr['pos'][0], tr['pos'][1], 0.0
                tr['history'].append(p)
                if len(tr['history']) > MAX_HISTORY:
                    tr['history'].pop(0)
        
        # build MarkerArray
        marr = MarkerArray()
        for tid, tr in self.tracks.items():
            if len(tr['history']) < 2 or tr['missed'] >= HIDE_AFTER:
                continue
            m = Marker()
            m.header.frame_id = fr
            m.header.stamp = stamp
            m.ns = 'people'
            m.id = tid
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.05
            
            # Assign color per ID
            color_idx = (tid - 1) % len(self.colors)
            r, g, b = self.colors[color_idx]
            m.color.a = 1.0
            m.color.r = r
            m.color.g = g
            m.color.b = b
            
            m.pose.orientation.w = 1.0
            m.points = tr['history']
            m.lifetime.sec = 0  # 0 => forever
            marr.markers.append(m)

        self.pub.publish(marr)

def main():
    rclpy.init()
    rclpy.spin(PersonMarkers())
    rclpy.shutdown()

if __name__ == '__main__':
    main()