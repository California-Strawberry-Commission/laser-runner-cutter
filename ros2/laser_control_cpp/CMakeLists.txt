cmake_minimum_required(VERSION 3.8)
project(laser_control_cpp)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(laser_control_interfaces REQUIRED)
find_package(spdlog REQUIRED)

###########
## Build ##
###########

include_directories(include)

# Add Helios DAC API shared library and Ether Dream shared library
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/lib/x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/lib/aarch64")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()
add_library(helios_dac_lib SHARED IMPORTED)
set_target_properties(helios_dac_lib PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libHeliosDacAPI.so")
add_library(ether_dream_dac_lib SHARED IMPORTED)
set_target_properties(ether_dream_dac_lib PROPERTIES IMPORTED_LOCATION "${LIB_PATH}/libEtherDream.so")

add_library(laser_dac
  src/laser_dac/helios.cpp
  src/laser_dac/ether_dream.cpp
)
target_link_libraries(laser_dac PRIVATE spdlog::spdlog)

add_executable(laser_control_node
  src/laser_control_node.cpp
)
target_link_libraries(laser_control_node PRIVATE laser_dac helios_dac_lib ether_dream_dac_lib)
ament_target_dependencies(laser_control_node PUBLIC rclcpp std_msgs std_srvs laser_control_interfaces)

#############
## Install ##
#############

install(TARGETS
  laser_control_node
  DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
