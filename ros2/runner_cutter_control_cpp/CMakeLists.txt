cmake_minimum_required(VERSION 3.8)
project(runner_cutter_control_cpp)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-unknown-pragmas)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(runner_cutter_control_interfaces REQUIRED)
find_package(laser_control_interfaces REQUIRED)
find_package(camera_control_interfaces REQUIRED)
find_package(spdlog REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)

###########
## Build ##
###########

include_directories(include)

# clients
add_library(clients SHARED
  src/clients/laser_control_client.cpp
  src/clients/camera_control_client.cpp
  src/clients/laser_detection_context.cpp
)
ament_target_dependencies(clients PUBLIC
  rclcpp
  std_srvs
  laser_control_interfaces
  camera_control_interfaces
)

# calibration
add_library(calibration SHARED
  src/calibration/calibration.cpp
  src/calibration/point_correspondences.cpp
)
target_link_libraries(calibration PUBLIC Eigen3::Eigen clients)

# tracking
add_library(tracking SHARED
  src/tracking/track.cpp
  src/tracking/tracker.cpp
)

# Node executable
add_executable(runner_cutter_control_node
  src/runner_cutter_control_node.cpp
)
target_link_libraries(runner_cutter_control_node PRIVATE clients calibration tracking fmt::fmt)
ament_target_dependencies(runner_cutter_control_node PUBLIC
  rclcpp
  std_msgs
  std_srvs
  runner_cutter_control_interfaces
  laser_control_interfaces
  camera_control_interfaces
)

# Tools
add_executable(test_calibration
  src/tools/test_calibration.cpp
)
target_link_libraries(test_calibration PRIVATE calibration)

#############
## Install ##
#############

install(TARGETS
  runner_cutter_control_node
  DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
