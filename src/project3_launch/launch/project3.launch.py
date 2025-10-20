import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Launch arguments
    bag_in = LaunchConfiguration('bag_in')
    bag_out = LaunchConfiguration('bag_out')

    return LaunchDescription([
        # Declare arguments
        DeclareLaunchArgument('bag_in', default_value='', description='Input bag file to play'),
        DeclareLaunchArgument('bag_out', default_value='output', description='Output bag file to record'),

        # Node: read_scan
        Node(
            package='read_scan',
            executable='read_scan_node',
            output='screen'
        ),

        # Node: publish_markers
        Node(
            package='publish_markers',
            executable='publish_markers_node',
            output='screen'
        ),

        # Play the input bag file
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', LaunchConfiguration('bag_in')],
            output='screen',
            shell=True
        ),

        # Record topics to the output bag file
        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'record',
                '-o', LaunchConfiguration('bag_out'),
                '/person_markers'
            ],
            output='screen',
            shell=True
        ),
    ])
