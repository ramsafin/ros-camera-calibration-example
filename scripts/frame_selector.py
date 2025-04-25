#!/usr/bin/env python3
# encoding: utf-8

import rospy
import rospkg
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import cv2
import numpy as np

import os
import uuid
import argparse
import threading
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Final, Dict, Any, Tuple, Optional


ROS_PACKAGE_PATH: Final[os.PathLike] = rospkg.RosPack().get_path("ros_camera_calibration_example")
DEFAULT_SAVE_DIR: Final[os.PathLike] = os.path.join(ROS_PACKAGE_PATH, "calib_data")

DEFAULT_BUF_SIZE: Final[int] = 30
DEFAULT_SELECT_TIMEOUT_SECS: Final[int] = 1


@dataclass
class Frame:
    image: np.ndarray
    timestamp: rospy.Time


@dataclass
class ChessboardSize:
    width: int
    height: int

    @property
    def tuple(self) -> Tuple[int, int]:
        return (self.width, self.height)


@dataclass
class ChessboardDetectionParams:
    size: ChessboardSize
    sharpness_threshold: int


@dataclass
class DatasetCollectionParams:
    buf_size: int
    num_samples: int
    timeout: rospy.Duration
    save_dir: os.PathLike


@dataclass
class Opts:
    chessboard: ChessboardDetectionParams
    collection: DatasetCollectionParams

@dataclass
class DatasetEntry:
    frame: Frame
    corners: np.ndarray


class Dataset:
    def __init__(self):
        self.entries: List[DatasetEntry] = []

    def save(self, filepath: os.PathLike):
        data_dict: Dict[str, Any] = {}
        for idx, e in enumerate(self.entries):
            data_dict[f"{idx}_image"]


class FrameSelector:
    def __init__(self, opts: Opts):
        self.opts = opts
        self.lock = threading.Lock()

        self.save_dir: os.PathLike = opts.collection.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.buffer: Deque[Frame] = deque(maxlen=self.opts.collection.buf_size)

        # список отобранных кадров
        self.selected_frames: List[DatasetEntry] = []

        # подписываемся на топик с изображениями
        rospy.Subscriber("image", Image, self._image_callback)

        # запускаем процесс отбора кадров по таймеру
        rospy.Timer(self.opts.collection.timeout, self._on_timer_callback)


    def save_dataset(self) -> None:
        if (not self.selected_frames):
            rospy.logwarn("Попытка сохранить пустой список кадров!")
            return False

        rospy.logwarn(f"Сохранение кадров в '{self.save_dir}'...")

        for image, timestamp in self.selected_frames:
            dt = datetime.fromtimestamp(timestamp.to_sec())
            filename: os.PathLike = f"{dt.strftime('%Y%m%d_%H%M%S.%f'[:-3])}.png"

            filepath: os.PathLike = os.path.join(self.save_dir, filename)

            try:
                cv2.imwrite(filepath, image)
                rospy.logdebug(f"Кадр сохранен: {filepath}")
            except Exception as e:
                rospy.logerr(f"Ошибка сохранения кадра: {filepath}. {e}")
                return False

        return True


    def _image_callback(self, msg: Image) -> None:
        try:
            frame = Frame(
                self.bridge.imgmsg_to_cv2(msg, "mono8"),
                msg.header.stamp
            )

            with self.lock:
                self.buffer.append(frame)

        except Exception as e:
            rospy.logerr(f"Ошибка обработки изображения: {e}")


    def get_latest_frame(self) -> Optional[Frame]:
        with self.lock:
            return self.buffer[-1] if self.buffer else None


    def _on_timer_callback(self, event: rospy.timer.TimerEvent) -> None:

        rospy.loginfo(f"Количество отобранных кадров: {len(self.selected_frames)}/{self.opts.num_samples}")

        # 1: проверка количества кадров
        if len(self.selected_frames) >= self.opts.num_samples:
            rospy.logwarn(f"Достигнуто максимальное количество кадров: {self.opts.num_samples}")
            rospy.signal_shutdown("Завершение по достижению лимита кадров")
            return

        # получение последнего кадра из буфера
        frame: Frame = self.get_latest_frame()

        if frame is None:
            rospy.logwarn(f"Пустой буфер кадров")
            return

        # 2: проверка размытости кадра
        if not self._is_sharp(frame.image):
            rospy.logwarn("Недостаточная четкость кадра")
            return

        # 3: проверка обнаруженных углов шахматной доски
        detected, corners = cv2.findChessboardCorners(
            frame.image, self.opts.chessboard_size.tuple, self.opts.chessboard_find_flags
        )

        if not detected:
            rospy.logwarn(f"Шахматная доска не обнаружена. Углы: {corners.shape}")
            return

        rospy.logwarn(f"Обнаружена шахматная доска. Углы: {corners.shape}")

        # уточняем координаты внутренних углов шахматной доски
        corners = cv2.cornerSubPix(frame.image, corners, self.opts.subpix_win_size, (-1, -1), self.opts.subpix_criteria)

        # добавляем кадр в список отобранных
        self.selected_frames.append(frame)


    def _is_sharp(self, image: np.ndarray) -> bool:
        return cv2.Laplacian(image, cv2.CV_64F).var() > self.opts.sharpness_threshold


def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description='Инструмент для сбора калибровочных кадров с шахматной доской'
    )
    parser.add_argument(
        '-s', '--samples',
        type=int, required=True,
        help='Количество необходимых калибровочных кадров'
    )
    parser.add_argument(
        '-x', '--width',
        type=int, required=True,
        help='Количество внутренних углов шахматной доски (по горизонтали)'
    )
    parser.add_argument(
        '-y', '--height',
        type=int, required=True,
        help='Количество внутренних углов шахматной доски (по вертикали)'
    )
    parser.add_argument(
        '-n', '--buf-size',
        type=int, default=DEFAULT_BUF_SIZE,
        help='Размер буфера кадров (по умолчанию: %(default)s)'
    )
    parser.add_argument(
        '-t', '--timeout',
        type=int, default=DEFAULT_SELECT_TIMEOUT_SECS,
        help='Интервал отбора кадров (секунды, по умолчанию: %(defaults)s)'
    )
    parser.add_argument(
        '-o', '--out',
        type=str, default=DEFAULT_SAVE_DIR,
        help='Директория для сохранения кадров (по умолчанию: %(default)s)'
    )
    parser.add_argument(
        '-w', '--subpix-win',
        type=int, default=3,
        help='Размер окна для cornerSubPix (по умолчанию: %(default)s)'
    )
    args, _ = parser.parse_known_args()
    return args


def args2opts(args: argparse.Namespace) -> Opts:
    return Opts(
        args.buf_size,
        args.samples,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1E-3),
        (5, 5), # sub-pixel window size
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK,
        rospy.Duration(args.timeout),
        ChessboardSize(args.width, args.height),
        10, # min angle diff
        args.out
    )


def main() -> None:
    rospy.init_node("frame_selector")
    opts = args2opts(parse_args())
    selector = FrameSelector(opts)
    rospy.spin()


if __name__ == "__main__":
    main()
