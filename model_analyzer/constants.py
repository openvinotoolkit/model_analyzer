# Copyright (C) 2019-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class YoloAnchors(Enum):
    YOLO_V2 = [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]
    TINY_YOLO_V2 = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
    YOLO_V3 = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
    YOLO_V4 = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    TINY_YOLO_V3_V4 = [10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]


class LayoutTypes(Enum):
    NCHW = ['N', 'C', 'H', 'W']
    NHWC = ['N', 'H', 'W', 'C']
    NC = ['N', 'C']
    CN = ['C', 'N']
    C = ['C']
