cmake_minimum_required(VERSION 3.8)
project(runner_cutter_control)

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

# Add matplotlib-cpp
set(MATPLOTLIB_DIR third_party/matplotlib-cpp)
add_library(matplotlib_cpp INTERFACE)
target_compile_features(matplotlib_cpp INTERFACE
  cxx_std_11
)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
target_link_libraries(matplotlib_cpp INTERFACE
  Python3::Python
  Python3::Module
)
target_include_directories(matplotlib_cpp INTERFACE
  ${MATPLOTLIB_DIR}
  ${Python3_INCLUDE_DIRS}
)
find_package(Python3 COMPONENTS NumPy)
if(Python3_NumPy_FOUND)
  target_link_libraries(matplotlib_cpp INTERFACE
    Python3::NumPy
  )
  target_include_directories(matplotlib_cpp INTERFACE
    ${Python3_NumPy_INCLUDE_DIRS}
  )
else()
  target_compile_definitions(matplotlib_cpp INTERFACE WITHOUT_NUMPY)
endif()

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

# prediction
add_library(prediction SHARED
  src/prediction/predictor.cpp
  src/prediction/last_known_predictor.cpp
  src/prediction/average_velocity_predictor.cpp
  src/prediction/kalman_filter_predictor.cpp
)
target_link_libraries(prediction PUBLIC Eigen3::Eigen)

# tracking
add_library(tracking SHARED
  src/tracking/track.cpp
  src/tracking/tracker.cpp
)
target_link_libraries(tracking PUBLIC prediction)

# Node executable
add_executable(runner_cutter_control_node
  src/runner_cutter_control_node.cpp
)
target_link_libraries(runner_cutter_control_node PRIVATE fmt::fmt clients calibration tracking prediction)
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

add_executable(test_predictor
  src/tools/test_predictor.cpp
)
target_link_libraries(test_predictor PRIVATE prediction matplotlib_cpp)

#############
## Install ##
#############

install(TARGETS
  runner_cutter_control_node
  test_calibration
  test_predictor
  DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
