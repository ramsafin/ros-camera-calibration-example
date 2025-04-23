cmake_minimum_required(VERSION 3.12)
project(ros_camera_calibration_example VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4 REQUIRED)

find_package(catkin REQUIRED
  rospy
  roscpp
  cv_bridge
  sensor_msgs
)

catkin_package()

add_executable(subscriber nodes/subscriber.cpp)
target_include_directories(subscriber PRIVATE ${catkin_INCLUDE_DIRS})
target_link_libraries(subscriber PRIVATE ${catkin_LIBRARIES} ${OpenCV_LIBS})
