cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(rgbd_bridge)

find_package(PCL REQUIRED)
find_package(OpenCV 2.4 REQUIRED)
#find_package(realsense)
#find_package(OpenNi2)

include(ExternalProject)

ExternalProject_Add(realsense1
  GIT_REPOSITORY https://github.com/IntelRealSense/librealsense.git
  GIT_TAG 66e42069837ed6e0eb46351cc4aa2acca49a4728
  CMAKE_ARGS
  -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX})

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_INSTALL_PREFIX}/include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})
  #${OPENNI2_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS}
  ${CMAKE_INSTALL_PREFIX}/lib)

add_library(rgbd_bridge SHARED
  src/rgbd_bridge.cc
  src/real_sense_sr300.cc)
  #src/openni.cc)
add_dependencies(rgbd_bridge realsense1)
target_link_libraries(rgbd_bridge
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  realsense)

install(TARGETS rgbd_bridge
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

install(DIRECTORY include/rgbd_bridge
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
