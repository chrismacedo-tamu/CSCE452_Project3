import rclpy

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('read_scan_node')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()