cmake_minimum_required(VERSION 3.8)
project(detection_cpp)

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

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(camera_control_interfaces REQUIRED)
find_package(camera_control_interfaces REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
# Force CMake to find the locally built OpenCV w/ CUDA support
set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(TensorRT REQUIRED)
set(Eigen3_DIR "third_party/eigen-3.3.9/cmake")
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

# Add CUDA libraries (Includes TensorRT)
set(CUDA_PATH "/usr/local/cuda-12.4")
link_directories(${CUDA_PATH}/lib64)
include_directories(${CUDA_PATH}/include)

cuda_add_library(fp16_utils src/tools/fp16_utils.cu)

# Add ByteTrack
include_directories(third_party/ByteTrack/include)
set(BYTETRACK_SOURCES
    third_party/ByteTrack/src/BYTETracker.cpp
    third_party/ByteTrack/src/kalmanFilter.cpp
    third_party/ByteTrack/src/lapjv.cpp
    third_party/ByteTrack/src/STrack.cpp
    third_party/ByteTrack/src/utils.cpp
)

add_library(bytetrack STATIC ${BYTETRACK_SOURCES})
target_include_directories(bytetrack PUBLIC 
    third_party/ByteTrack/include
    ${EIGEN3_INCLUDE_DIR}
)

# Add BS Thread Pool
include_directories(third_party/bs_thread_pool/include)

# Node executable
add_executable(detection_node
  src/detection_node.cpp
)
target_link_libraries(detection_node PRIVATE 
  fmt::fmt 
  spdlog::spdlog 
  ${ARENA_LIBS} 
  ${OpenCV_LIBS} 
  ${TensorRT_LIBRARIES} 
  nvinfer 
  nvonnxparser 
  cudart 
  fp16_utils 
  bytetrack
)
ament_target_dependencies(detection_node PUBLIC
  rclcpp
  std_msgs
  std_srvs
  cv_bridge
  OpenCV
  camera_control_interfaces
)

# Tools
add_executable(test_singleFrame src/tools/test_singleFrame.cpp)
target_include_directories(test_singleFrame PRIVATE ${TensorRT_INCLUDE_DIRS})
target_link_libraries(test_singleFrame PRIVATE spdlog::spdlog ${ARENA_LIBS} ${OpenCV_LIBS} ${TensorRT_LIBRARIES} fmt::fmt nvinfer nvonnxparser cudart)
ament_target_dependencies(test_singleFrame PUBLIC rclcpp std_msgs OpenCV)

add_executable(test_loop src/tools/test_loop.cpp)
target_include_directories(test_loop PRIVATE ${TensorRT_INCLUDE_DIRS})
target_link_libraries(test_loop PRIVATE spdlog::spdlog ${ARENA_LIBS} ${OpenCV_LIBS} ${TensorRT_LIBRARIES} fmt::fmt nvinfer nvonnxparser cudart fp16_utils bytetrack)
ament_target_dependencies(test_loop PUBLIC rclcpp std_msgs OpenCV)

#############
## Install ##
#############

install(TARGETS
  detection_node
  test_singleFrame
  test_loop
  DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
