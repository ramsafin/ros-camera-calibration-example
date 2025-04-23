# opencv-ros-template
ROS project template for OpenCV related applications.

## Getting Started

Start fake images publisher node:
```shell
rosrun template publisher.py image:=image_raw
```

Subscribers:

Python
```shell
rosrun template subscriber.py image:=image_raw
```

C++
```shell
rosrun template subscriber image:=image_raw
```
