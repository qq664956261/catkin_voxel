cmake_minimum_required(VERSION 3.0.2)
project(voxel_slam)

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "-std=c++14")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

ADD_COMPILE_OPTIONS(-std=c++17 )
set(CMAKE_CXX_FLAGS "-std=c++17 -O3" )

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -std=c++0x -std=c++17 -fexceptions")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  pcl_conversions
  pcl_ros
  livox_ros_driver
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(TBB REQUIRED)
#find_package(GTSAM REQUIRED QUIET)
# find_package(Ceres REQUIRED)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs 
  DEPENDS EIGEN3 PCL 
  INCLUDE_DIRS
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
# ${CERES_INCLUDE_DIRS}
  #${GTSAM_INCLUDE_DIR}
)

add_executable(voxelslam src/voxelslam.cpp )
target_link_libraries(voxelslam ${catkin_LIBRARIES} ${PCL_LIBRARIES} TBB::tbb)







