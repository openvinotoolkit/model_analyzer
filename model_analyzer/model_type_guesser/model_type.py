# Copyright (C) 2019-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from enum import Enum


class ModelType(Enum):
    YOLO = 'yolo'
    YOLO_V2 = 'yolo_v2'
    TINY_YOLO_V2 = 'tiny_yolo_v2'
    YOLO_V3 = 'yolo_v3'
    YOLO_V4 = 'yolo_v4'
    TINY_YOLO_V3_V4 = 'tiny_yolo_v3_v4'
    SSD = 'ssd'
    CLASSIFICATION = 'classificator'
    SEMANTIC_SEGM = 'segmentation'
    INSTANCE_SEGM = 'mask_rcnn'
    IN_PAINTING = 'inpainting'
    STYLE_TRANSFER = 'style_transfer'
    SUPER_RESOLUTION = 'super_resolution'
    FACE_RECOGNITION = 'face_recognition'
    LANDMARK_DETECTION = 'landmark_detection'
    GENERIC = 'generic'
