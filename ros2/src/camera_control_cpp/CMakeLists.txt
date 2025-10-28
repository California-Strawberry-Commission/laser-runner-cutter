cmake_minimum_required(VERSION 3.8)
project(camera_control_cpp)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-fpermissive -Wall -Wextra -Wpedantic -Wno-unknown-pragmas)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(camera_control_interfaces REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)

###########
## Build ##
###########

include_directories(include)

# Add Arena SDK
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(ARENA_SDK_PATH "/opt/ArenaSDK/ArenaSDK_Linux_x64")
    link_directories(
      ${ARENA_SDK_PATH}/lib64
      ${ARENA_SDK_PATH}/GenICam/library/lib/Linux64_x64
      ${ARENA_SDK_PATH}/ffmpeg
    )
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(ARENA_SDK_PATH "/opt/ArenaSDK/ArenaSDK_Linux_ARM64")
    link_directories(
      ${ARENA_SDK_PATH}/lib
      ${ARENA_SDK_PATH}/GenICam/library/lib/Linux64_ARM
      ${ARENA_SDK_PATH}/ffmpeg
    )
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()
set(GENICAM_LIBS
  GCBase_gcc54_v3_3_LUCID
  GenApi_gcc54_v3_3_LUCID
  Log_gcc54_v3_3_LUCID
  log4cpp_gcc54_v3_3_LUCID
  MathParser_gcc54_v3_3_LUCID
  NodeMapData_gcc54_v3_3_LUCID
  XmlParser_gcc54_v3_3_LUCID
)
set(FFMPEG_LIBS avcodec avformat avutil swresample)
set(ARENA_LIBS arena save gentl ${GENICAM_LIBS} ${FFMPEG_LIBS} pthread lucidlog)
set(ARENA_INCLUDE_DIRS
  ${ARENA_SDK_PATH}/include/Arena
  ${ARENA_SDK_PATH}/include/Save
  ${ARENA_SDK_PATH}/include/GenTL
  ${ARENA_SDK_PATH}/GenICam/library/CPP/include
)

# Add BS Thread Pool
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/bs_thread_pool/include)

# Add CLI11
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/CLI11 ${CMAKE_CURRENT_BINARY_DIR}/cli11-build EXCLUDE_FROM_ALL)

# camera
add_library(camera SHARED
  src/camera/calibration.cpp
  src/camera/lucid_camera.cpp
)
target_include_directories(camera PUBLIC
  ${ARENA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(camera PRIVATE spdlog::spdlog ${ARENA_LIBS} ${OpenCV_LIBS})
ament_target_dependencies(camera PUBLIC
  rclcpp
  sensor_msgs
)

# utils
add_library(utils SHARED
  src/utils/rgbd_alignment.cpp
)
target_include_directories(utils PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(utils PRIVATE
  spdlog::spdlog
  ${OpenCV_LIBS}
)

# Composable node
add_library(camera_control_node_composable SHARED
  src/camera_control_node.cpp
)
target_link_libraries(camera_control_node_composable PRIVATE fmt::fmt Eigen3::Eigen camera)
ament_target_dependencies(camera_control_node_composable PUBLIC
  rclcpp
  rclcpp_components
  std_msgs
  std_srvs
  geometry_msgs
  tf2
  tf2_ros
  camera_control_interfaces
  cv_bridge
  OpenCV
)
rclcpp_components_register_node(
  camera_control_node_composable
  PLUGIN "CameraControlNode"
  EXECUTABLE camera_control_node
)

# Tools
add_executable(latency_clock src/tools/latency_clock.cpp)

add_executable(lucid_calibrate src/tools/lucid_calibrate.cpp)
target_link_libraries(lucid_calibrate PRIVATE camera CLI11::CLI11 spdlog::spdlog ${OpenCV_LIBS})

#############
## Install ##
#############

# Install lib + headers that should be available to other packages
install(TARGETS utils
  EXPORT export_camera_control_cpp_targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)

# Export the targets with a namespace so downstream sees camera_control_cpp::utils
install(EXPORT export_camera_control_cpp_targets
  NAMESPACE camera_control_cpp::
  DESTINATION share/${PROJECT_NAME}/cmake
)

# ament export metadata
ament_export_targets(export_camera_control_cpp_targets HAS_LIBRARY_TARGET)
ament_export_include_directories(include)

# Install composable node
install(TARGETS camera_control_node_composable
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install tools
install(TARGETS
  latency_clock
  lucid_calibrate
  DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
