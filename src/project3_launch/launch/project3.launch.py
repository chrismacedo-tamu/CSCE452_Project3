import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, RegisterEventHandler, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    # Arguments for Launch
    bag_in = LaunchConfiguration('bag_in')
    bag_out = LaunchConfiguration('bag_out')

    # read_scan Node
    read_scan_node = Node(
        package='read_scan',
        executable='read_scan_node',
        name='read_scan_node',
        output='screen'
    )

    # publish_markers Node
    publish_markers_node = Node(
        package='publish_markers',
        executable='publish_markers_node',
        name='publish_markers_node',
        output='screen'
    )
    
    # Laser Scan bag playback
    bag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag_in],
        output='screen'
    )

    # Recording all topics
    bag_record = TimerAction(
        period=2.0,
        actions=[
            ExecuteProcess(cmd=['ros2', 'bag', 'record', '-a', '-o', bag_out],
            output='screen')
        ]
    )

    # Shutting evertything down when the bag playing finishes
    stop = RegisterEventHandler(
        event_handler = OnProcessExit(
            target_action=bag_play,
            on_exit = [
                LogInfo(msg='Bag playback finished. Stopped recording...'),
                ExecuteProcess(cmd=['pkill', '-f', 'ros2 bag record']),
                ExecuteProcess(cmd=['pkill', '-f', 'read_scan_node']),
                ExecuteProcess(cmd=['pkill', '-f', 'publish_markers_node']),
            ],
        )
    )


    return LaunchDescription([
        # Declare arguments
        DeclareLaunchArgument('bag_in', default_value='', description='Input bag file to play'),
        DeclareLaunchArgument('bag_out', default_value='output', description='Output bag file to record'),


        read_scan_node,
        LogInfo(msg='read_scan_node started...'), 
        publish_markers_node,
        LogInfo(msg='publish_markers_node started...'), 
        bag_play,
        LogInfo(msg='bag_play started...'), 
        bag_record,
        LogInfo(msg='record_bag started...'), 
        stop,
        LogInfo(msg='stop_recording started...') 
    ])
