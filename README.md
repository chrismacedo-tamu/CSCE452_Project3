# CSCE452_Project3

## Run using launch files:
- cd CSCE452_Project3
- colcon build
- source install/setup.bash
- ros2 launch project3_launch project3.launch.py bag_in:=project3-bags/01 bag_out:=output

## Visualizing in rviz2
- Set fixed frame to "laser"
- Add LaserScan
- - Set Topic to "/scan"
- Add MarkerArray
- - Set Topic to "/people_markers"

