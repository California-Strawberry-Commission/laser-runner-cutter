cmake_minimum_required(VERSION 3.8)
project(tensorrt_cpp)

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

# Tools
add_executable(detect_laser
  src/tools/detect_laser.cpp
)
target_include_directories(detect_laser PUBLIC ${YOLOV8_TRT_INCLUDE_DIRS})
target_link_libraries(detect_laser PRIVATE detector spdlog::spdlog yolov8_trt)

#############
## Install ##
#############

ament_package()