cmake_minimum_required(VERSION 3.11)
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
find_package(camera_control_interfaces REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)

include(FetchContent)

# CLI11 (header-only)
FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.5.0
)
FetchContent_MakeAvailable(CLI11)

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
include_directories(third_party/bs_thread_pool/include)

# camera
add_library(camera SHARED
  src/camera/calibration.cpp
  src/camera/lucid_camera.cpp
  src/camera/lucid_frame.cpp
)
target_include_directories(camera PUBLIC
  ${ARENA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(camera PRIVATE spdlog::spdlog ${ARENA_LIBS} ${OpenCV_LIBS})

install(TARGETS camera
        EXPORT export_camera_control_node_component
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
)

# Node executable
# add_executable(camera_control_node
#   src/camera_control_node.cpp
# )
add_library(camera_control_node_component SHARED
  src/camera_control_node.cpp
)
target_link_libraries(camera_control_node_component PRIVATE fmt::fmt camera)
ament_target_dependencies(camera_control_node_component PUBLIC
  rclcpp
  rclcpp_components
  std_msgs
  std_srvs
  camera_control_interfaces
  cv_bridge
  OpenCV
)
rclcpp_components_register_node(
    camera_control_node_component
    PLUGIN "CameraControlNode"
    EXECUTABLE camera_control_node
)

# ament_export_targets(export_camera_control_node_component)
install(TARGETS camera_control_node_component
        EXPORT export_camera_control_node_component
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
)

install(EXPORT export_camera_control_node_component
  FILE export_camera_control_node_component.cmake
  NAMESPACE camera_control_cpp::
  DESTINATION share/${PROJECT_NAME}/cmake
)
ament_export_targets(export_camera_control_node_component)

# Tools
add_executable(lucid_calibrate src/tools/lucid_calibrate.cpp)
target_link_libraries(lucid_calibrate PRIVATE camera CLI11::CLI11 spdlog::spdlog ${OpenCV_LIBS})

#############
## Install ##
#############

install(TARGETS
  # camera_control_node
  lucid_calibrate
  DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
