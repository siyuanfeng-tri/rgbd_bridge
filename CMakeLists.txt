cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(rgbd_bridge)

find_package(PCL REQUIRED)
find_package(OpenCV 2.4 REQUIRED)
find_package(realsense2 REQUIRED)

include(ExternalProject)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "-g -O2 -Wall ${CMAKE_CXX_FLAGS}")

#ExternalProject_Add(realsense1
#  GIT_REPOSITORY https://github.com/IntelRealSense/librealsense.git
#  GIT_TAG 0cc659d
#  CMAKE_ARGS
#  -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX})

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_INSTALL_PREFIX}/include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS}
  ${CMAKE_INSTALL_PREFIX}/lib)

add_library(rgbd_bridge SHARED
  src/rgbd_bridge.cc
  src/real_sense_sr300.cc)
target_link_libraries(rgbd_bridge
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  realsense2)

add_executable(test_real_sense src/test_real_sense.cc)
target_link_libraries(test_real_sense rgbd_bridge)

install(TARGETS rgbd_bridge
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

install(DIRECTORY include/rgbd_bridge
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

install(TARGETS test_real_sense
  DESTINATION ${CMAKE_INSTALL_PREFIX})
