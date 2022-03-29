# Copyright (C) 2019-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from model_analyzer.model_metadata.model_metadata import ModelMetaData
from model_analyzer.model_type_analyzer.model_type import ModelType
from model_analyzer.model_type_analyzer.model_type_analyzer import (
    YoloV2TypeAnalyzer, TinyYoloV2TypeAnalyzer, YoloV3TypeAnalyzer, YoloV4TypeAnalyzer, TinyYoloV3V4TypeAnalyzer,
    GenericYoloTypeAnalyzer, SSDTypeAnalyzer, ClassificationTypeAnalyzer, InstanceSegmentationTypeAnalyzer,
    SemanticSegmentationTypeAnalyzer, InPaintingTypeAnalyzer, StyleTransferTypeAnalyzer, SuperResolutionTypeAnalyzer,
    FaceRecognitionTypeAnalyzer, LandmarkDetectionTypeAnalyzer, GenericModelTypeAnalyzer
)


class ModelTypeGuesser:
    @staticmethod
    def get_model_type(model_metadata: ModelMetaData) -> ModelType:
        ordered_model_type_analyzers = (
            YoloV2TypeAnalyzer, TinyYoloV2TypeAnalyzer,
            YoloV3TypeAnalyzer, YoloV4TypeAnalyzer,
            TinyYoloV3V4TypeAnalyzer, GenericYoloTypeAnalyzer,
            SSDTypeAnalyzer, ClassificationTypeAnalyzer,
            InstanceSegmentationTypeAnalyzer, SemanticSegmentationTypeAnalyzer,
            InPaintingTypeAnalyzer, StyleTransferTypeAnalyzer,
            SuperResolutionTypeAnalyzer, FaceRecognitionTypeAnalyzer,
            LandmarkDetectionTypeAnalyzer,
            GenericModelTypeAnalyzer
        )

        for type_analyzer_class in ordered_model_type_analyzers:
            if type_analyzer_class.is_like(model_metadata):
                return type_analyzer_class.get_type()
        raise AssertionError
