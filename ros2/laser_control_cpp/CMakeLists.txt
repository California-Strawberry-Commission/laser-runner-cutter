cmake_minimum_required(VERSION 3.8)
project(laser_control_cpp)

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
find_package(std_srvs REQUIRED)
find_package(laser_control_interfaces REQUIRED)
find_package(spdlog REQUIRED)
# Find libusb
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBUSB REQUIRED libusb-1.0)

###########
## Build ##
###########

include_directories(include)

# Add Helios DAC SDK
set(HELIOS_DAC_SDK_DIR third_party/helios_dac/sdk)
add_library(helios_dac_sdk SHARED
  ${HELIOS_DAC_SDK_DIR}/HeliosDac.cpp
)
target_include_directories(helios_dac_sdk PUBLIC
  ${HELIOS_DAC_SDK_DIR}
)
target_link_libraries(helios_dac_sdk PRIVATE ${LIBUSB_LIBRARIES})
target_compile_options(helios_dac_sdk PRIVATE -fpermissive -Wno-write-strings -Wno-pedantic -Wno-unused-variable -Wno-address)

# Add Ether Dream SDK
set(ETHER_DREAM_SDK_DIR third_party/ether-dream-sdk/src)
add_library(ether_dream_sdk SHARED
  ${ETHER_DREAM_SDK_DIR}/etherdream.c
)
target_include_directories(ether_dream_sdk PUBLIC
  ${ETHER_DREAM_SDK_DIR}
)

# dacs
add_library(dacs SHARED
  src/dacs/helios.cpp
  src/dacs/ether_dream.cpp
)
target_include_directories(dacs PUBLIC
  ${HELIOS_DAC_SDK_DIR}
  ${ETHER_DREAM_SDK_DIR}
)
target_link_libraries(dacs PRIVATE helios_dac_sdk ether_dream_sdk spdlog::spdlog)

# Node executable
add_executable(laser_control_node
  src/laser_control_node.cpp
)
target_link_libraries(laser_control_node PRIVATE dacs)
ament_target_dependencies(laser_control_node PUBLIC rclcpp std_srvs laser_control_interfaces)

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
