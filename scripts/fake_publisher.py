#!/usr/bin/env python3
# encoding: utf-8

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import cv2
import numpy as np

from typing import Final

ROS_PARAM_PUB_RATE: Final[int] = 1
ROS_IMAGE_TOPIC: Final[str] = "image"


def generate_digit_image(digit: int) -> np.ndarray:
    """Генерирует изображение цифры (0-9)"""
    assert 0 <= digit <= 9, "Цифры должны быть в диапазоне от 0 до 9!"
    img = np.zeros((512, 512, 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(str(digit), font, 8, 10)[0]
    text_x = (512 - text_size[0]) // 2
    text_y = (512 + text_size[1]) // 2
    cv2.putText(img, str(digit), (text_x, text_y), font, 8, (255, 255, 255), 10)

    return img

def main() -> None:
  rospy.init_node("fake_publisher")

  pub_frequency: int = rospy.get_param("~rate", ROS_PARAM_PUB_RATE)

  publisher = rospy.Publisher(ROS_IMAGE_TOPIC, Image, queue_size=30)  # попробуйте queue_size=None
  rospy.loginfo(f"Публикуем в '{rospy.resolve_name(ROS_IMAGE_TOPIC)}' с частотой {pub_frequency} Гц ...")

  current_digit = 0
  bridge = CvBridge()
  rate = rospy.Rate(pub_frequency)

  while not rospy.is_shutdown():
    image = generate_digit_image(current_digit)
    publisher.publish(bridge.cv2_to_imgmsg(image, encoding="bgr8"))
    current_digit = (current_digit + 1) % 10
    rate.sleep()


if __name__ == '__main__':
    main()
