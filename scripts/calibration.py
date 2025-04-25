#!/usr/bin/env python3

import rospy
import rospkg

import os
import glob
import argparse
from dataclasses import dataclass
from typing import Final, Dict, Any, Tuple, List

import cv2
import numpy as np

ROS_PACKAGE_PATH: Final[os.PathLike] = rospkg.RosPack().get_path("ros_camera_calibration_example")
DEFAULT_IMAGE_DIR: Final[os.PathLike] = os.path.join(ROS_PACKAGE_PATH, "calib_data")


@dataclass
class ChessboardSize:
    width: int
    height: int

    @property
    def tuple(self) -> Tuple[int, int]:
        return (self.width, self.height)


def get_object_point(board_size: ChessboardSize) -> np.ndarray:
    object_point = np.zeros((1, board_size.height * board_size.width, 3), np.float32)
    object_point[0, :, :2] = np.mgrid[0:board_size.height, 0:board_size.width].T.reshape(-1, 2)
    return object_point


def retrieve_image_filepaths(image_dir: os.PathLike, ext: str = '*.png') -> List[os.PathLike]:
    images = glob.glob(os.path.join(image_dir, ext))
    rospy.loginfo(f"Найдено {len(images)} изображений в '{image_dir}'")
    return images


def read_images(image_dir: os.PathLike, ext: str = '*.png') -> List[np.ndarray]:
    images: List[np.ndarray] = []
    image_filepaths: List[os.PathLike] = retrieve_image_filepaths(image_dir, ext)

    for filepath in image_filepaths:
        images.append(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)) # можно и в цветном формате

    return images


def parse_args() -> Dict[str, Any]:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-x', '--width', type=int, help='количество внутренних углов (по горизонтали)', required=True)
    arg_parser.add_argument('-y', '--height', type=int, help='количество внутренних углов (по вертикали)', required=True)
    arg_parser.add_argument('-i', '--in', type=str, help='директория с калибровочными изображениями', default=DEFAULT_IMAGE_DIR)
    return vars(arg_parser.parse_args())


def main() -> None:
    args: Dict[str, Any] = parse_args()

    image_dir: os.PathLike = args['in']
    board_size = ChessboardSize(args['width'], args['height'])

    for index in range(args['from'], args['to'] + 1):
        calib_params['index'] = index
        print('Calibration params:\n{}'.format(calib_params))

        dataset_path = resolve_dataset_path(calib_params)

        print('Reading images from {}'.format(dataset_path))
        images = read_images(dataset_path)

        object_point = get_object_point(board_size)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1E-3)
        find_flags = np.sum([cv2.CALIB_CB_ADAPTIVE_THRESH,
                             cv2.CALIB_CB_NORMALIZE_IMAGE,
                             cv2.CALIB_CB_FAST_CHECK], dtype=np.uint32)

        image_pts = []
        object_pts = []

        print('Finding chessboard corners ...')

        start_time = time.time()

        for idx, image in enumerate(images):
            detected, corners = cv2.findChessboardCorners(image, board_size, flags=find_flags)

            if detected:
                corners_refined = cv2.cornerSubPix(image, corners, (5, 5), (-1, -1), criteria)
                image_pts.append(corners_refined)
                object_pts.append(object_point)
            else:
                print('Image {}: no corners found.'.format(idx))

        print('Calibration samples found: {}'.format(len(image_pts)))

        if len(image_pts) > 15:
            print('Estimating camera parameters ...')
            err, P, D, R, t = cv2.calibrateCamera(object_pts, image_pts, images[0].shape[::-1],
                                                  None, None, criteria=criteria)

            eval_time = time.time() - start_time

            print('Repr. error (RMSE): {} pixels'.format(err))
            save_calibration(dataset_path, P, D, R, t, err, eval_time)
        else:
            print('Not enough calibration samples for calibration.')

    print('Done!')


if __name__ == '__main__':
    main()
