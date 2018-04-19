cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(rgbd_bridge)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(realsense2 REQUIRED)

set (CMAKE_CXX_STANDARD 14)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_INSTALL_PREFIX}/include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})

add_library(rgbd_bridge SHARED
  src/intrinsics.cc
  src/rgbd_bridge.cc
  src/real_sense_d400.cc
  src/real_sense_common.cc)

target_link_libraries(rgbd_bridge
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  realsense2)

add_executable(test_real_sense
  src/test_real_sense.cc
  src/rgbd_display_app.cc)
target_link_libraries(test_real_sense rgbd_bridge)
