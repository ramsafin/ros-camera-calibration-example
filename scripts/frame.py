#!/usr/bin/env python3
# encoding: utf-8

import rospy
import numpy as np
from dataclasses import dataclass

@dataclass
class Frame:
    image: np.ndarray
    timestamp: rospy.Time
