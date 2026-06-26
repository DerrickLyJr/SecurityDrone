cmake_minimum_required(VERSION 3.12.2)
project(bluerov2_gz)

find_package(ament_cmake_auto REQUIRED)

# Update GZ_SIM_RESOURCE_PATH
# Reference: https://colcon.readthedocs.io/en/released/developer/environment.html#dsv-files
ament_environment_hooks(hooks/setup.dsv)

ament_auto_find_build_dependencies()

find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

add_executable(teleop_node scripts/teleop_node.cpp)
ament_target_dependencies(teleop_node rclcpp std_msgs geometry_msgs)

find_package(sensor_msgs REQUIRED)

add_executable(imu_listener scripts/imu_listener.cpp)
ament_target_dependencies(imu_listener rclcpp sensor_msgs geometry_msgs)

# --- C++ ports of performance-sensitive Python nodes (Python originals kept side by side) ---
find_package(Eigen3 REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(ament_index_cpp REQUIRED)

add_executable(ekf_9dof_localizer_node scripts/ekf_9dof_localizer.cpp)
target_include_directories(ekf_9dof_localizer_node PRIVATE ${EIGEN3_INCLUDE_DIR})
ament_target_dependencies(ekf_9dof_localizer_node
  rclcpp sensor_msgs geometry_msgs nav_msgs tf2_ros)

add_executable(stereo_vision_node scripts/stereo_vision.cpp)
target_include_directories(stereo_vision_node PRIVATE ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(stereo_vision_node
  rclcpp sensor_msgs cv_bridge message_filters OpenCV ament_index_cpp)

# Install launch files from the launch directory to the share directory
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY scripts/computer_vision/
  DESTINATION lib/${PROJECT_NAME}/computer_vision
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY models worlds
  DESTINATION share/${PROJECT_NAME}
)

# Install Python scripts
install(PROGRAMS
  scripts/depth_calculator.py
  scripts/stereo_vision.py
  scripts/ekf_9dof_localizer.py
  scripts/localization_master.py
  scripts/flight_controller.py
  scripts/pressure_bridge.py
  scripts/ground_truth_bridge.py
  scripts/localization_validator.py
  scripts/VisionLocalMath.py
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  teleop_node
  imu_listener
  ekf_9dof_localizer_node
  stereo_vision_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_auto_package()