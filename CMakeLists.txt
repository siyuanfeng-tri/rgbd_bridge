cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(rgbd_bridge)

find_package(PCL REQUIRED)
find_package(OpenCV 2.4 REQUIRED)
find_package(realsense)
#find_package(OpenNi2)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})
  #${OPENNI2_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})

add_definitions(
  -std=c++1y -g -Wall -O2 -fPIC
  ${PCL_DEFINITIONS})

add_library(rgbd_bridge SHARED
  src/rgbd_bridge.cc
  src/real_sense_sr300.cc)
  #src/openni.cc)
target_link_libraries(rgbd_bridge
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  realsense::realsense)

install(TARGETS rgbd_bridge
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

install(DIRECTORY include/rgbd_bridge
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
