cmake_minimum_required(VERSION 3.8)
project(detection)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-fpermissive -Wall -Wextra -Wpedantic -Wno-unknown-pragmas)
endif()

# Add CMake module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

option(ENABLE_BENCHMARKS "Enable benchmark code" OFF)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(camera_control_interfaces REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
# Use locally built OpenCV with CUDA support
set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(cv_bridge REQUIRED)

###########
## Build ##
###########

include_directories(include)

# Add CLI11
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/CLI11 ${CMAKE_CURRENT_BINARY_DIR}/cli11-build EXCLUDE_FROM_ALL)

# Add ByteTrack-cpp
set(BYTETRACK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/ByteTrack-cpp)
set(BYTETRACK_INCLUDE_DIRS
  ${BYTETRACK_DIR}/include
)
add_library(bytetrack SHARED
  ${BYTETRACK_DIR}/src/BYTETracker.cpp
  ${BYTETRACK_DIR}/src/KalmanFilter.cpp
  ${BYTETRACK_DIR}/src/lapjv.cpp
  ${BYTETRACK_DIR}/src/Object.cpp
  ${BYTETRACK_DIR}/src/Rect.cpp
  ${BYTETRACK_DIR}/src/STrack.cpp
)
target_include_directories(bytetrack PUBLIC
  ${BYTETRACK_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)
target_link_libraries(bytetrack PRIVATE Eigen3::Eigen)

# detector
add_library(detector SHARED
  src/detector/yolov8.cpp
  src/detector/runner_detector.cpp
)
target_include_directories(detector PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${BYTETRACK_INCLUDE_DIRS}
)
target_link_libraries(detector
  PUBLIC
    bytetrack
  PRIVATE
    spdlog::spdlog
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
)
ament_target_dependencies(detector PUBLIC
  ament_index_cpp
)
if(ENABLE_BENCHMARKS)
  target_compile_definitions(detector PRIVATE ENABLE_BENCHMARKS)
endif()

# Composable node
add_library(detection_node_composable SHARED
  src/detection_node.cpp
)
target_link_libraries(detection_node_composable PRIVATE detector fmt::fmt ${OpenCV_LIBS})
ament_target_dependencies(detection_node_composable PUBLIC
  rclcpp
  rclcpp_components
  std_msgs
  std_srvs
  sensor_msgs
  cv_bridge
)
rclcpp_components_register_node(
  detection_node_composable
  PLUGIN "DetectionNode"
  EXECUTABLE detection_node
)

add_library(test_image_publisher_composable SHARED
  src/test_image_publisher.cpp
)
ament_target_dependencies(test_image_publisher_composable PUBLIC
  rclcpp
  rclcpp_components
  std_msgs
  std_srvs
  sensor_msgs
  OpenCV
)
rclcpp_components_register_node(
  test_image_publisher_composable
  PLUGIN "TestImagePublisherNode"
  EXECUTABLE test_image_publisher
)

# Tools
add_executable(test_yolov8
  src/tools/test_yolov8.cpp
)
target_link_libraries(test_yolov8 PRIVATE CLI11::CLI11 detector ${OpenCV_LIBS})

add_executable(test_runner_detector
  src/tools/test_runner_detector.cpp
)
target_link_libraries(test_runner_detector PRIVATE CLI11::CLI11 detector ${OpenCV_LIBS})

#############
## Install ##
#############

# Install composable node
install(TARGETS
  detection_node_composable
  test_image_publisher_composable
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install tools
install(TARGETS
  test_yolov8
  test_runner_detector
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install model files
install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
