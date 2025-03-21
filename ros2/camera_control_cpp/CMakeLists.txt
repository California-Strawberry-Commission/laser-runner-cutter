cmake_minimum_required(VERSION 3.8)
project(camera_control_cpp)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-deprecated-declarations -Wno-unknown-pragmas)
endif()

# Add CMake module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(camera_control_interfaces REQUIRED)
find_package(spdlog REQUIRED)
# Use locally built OpenCV with CUDA support
set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)
find_package(fmt REQUIRED)

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

# Add YOLOv8-TensorRT
set(TENSORRT_CPP_API_DIR third_party/tensorrt-cpp-api)
add_library(tensorrt_cpp_api SHARED
  ${TENSORRT_CPP_API_DIR}/src/engine.cpp)
target_include_directories(tensorrt_cpp_api PUBLIC
  ${TENSORRT_CPP_API_DIR}/include
  ${TENSORRT_CPP_API_DIR}/include/interfaces
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TensorRT_INCLUDE_DIRS}
)
target_link_libraries(tensorrt_cpp_api PUBLIC
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
  ${TensorRT_LIBRARIES}
  fmt::fmt
)

set(YOLOV8_TRT_DIR third_party/YOLOv8-TensorRT-CPP)
add_library(yolov8_trt SHARED
  ${YOLOV8_TRT_DIR}/src/yolov8.cpp)
target_link_libraries(yolov8_trt PUBLIC tensorrt_cpp_api ${OpenCV_LIBS})
target_include_directories(yolov8_trt PUBLIC ${TENSORRT_CPP_API_DIR}/src)
set(YOLOV8_TRT_INCLUDE_DIRS ${YOLOV8_TRT_DIR}/src)

# Camera
add_library(camera
  src/camera/lucid_camera.cpp
  src/camera/lucid_frame.cpp
)
target_include_directories(camera PUBLIC
  ${ARENA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(camera PRIVATE spdlog::spdlog ${ARENA_LIBS} ${OpenCV_LIBS})

# Detector
add_library(detector
  src/detector/circle_detector.cpp
  src/detector/laser_detector.cpp
)
target_include_directories(detector PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TensorRT_INCLUDE_DIRS}
)
target_link_libraries(detector PRIVATE spdlog::spdlog ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES})

# Node
add_executable(camera_control_node
  src/camera_control_node.cpp
)
target_link_libraries(camera_control_node PRIVATE camera detector)
ament_target_dependencies(camera_control_node PUBLIC
  rclcpp
  std_msgs
  std_srvs
  camera_control_interfaces
  cv_bridge
  OpenCV
)

# Tools
add_executable(detect_circle
  src/tools/detect_circle.cpp
)
target_link_libraries(detect_circle PRIVATE detector)

add_executable(detect_laser
  src/tools/detect_laser.cpp
)
target_include_directories(detect_laser PUBLIC ${YOLOV8_TRT_INCLUDE_DIRS})
target_link_libraries(detect_laser PRIVATE detector spdlog::spdlog yolov8_trt)

#############
## Install ##
#############

install(TARGETS
  camera_control_node
  DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
