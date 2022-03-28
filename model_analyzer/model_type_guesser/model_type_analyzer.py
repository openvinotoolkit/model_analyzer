# Copyright (C) 2019-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, List

from model_analyzer.constants import YoloAnchors
from model_analyzer.layout_utils import parse_node_layout
from model_analyzer.model_metadata.model_metadata import ModelMetaData
from model_analyzer.model_type_guesser.model_type import ModelType
from model_analyzer.shape_utils import get_shape_for_node_safely


class GenericModelTypeAnalyzer:
    _topology_type = ModelType.GENERIC

    def __init__(self, model_metadata: ModelMetaData):
        self._model_metadata = model_metadata

    @classmethod
    def is_like(cls, unused_model_metadata: ModelMetaData) -> bool:
        """
        Need to overwrite in each child class
        """
        return True

    @classmethod
    def get_type(cls) -> ModelType:
        return cls._topology_type

    @property
    def model_metadata(self) -> ModelMetaData:
        return self._model_metadata

    @staticmethod
    def _all_outputs_are_images(model_metadata: ModelMetaData) -> bool:
        return all(
            len(get_shape_for_node_safely(output)) == 4
            for output in model_metadata.outputs
        )

    @staticmethod
    def _check_single_image_input(model_metadata: ModelMetaData) -> bool:
        if not len(model_metadata.inputs) == 1:
            return False
        model_input = model_metadata.inputs[0]
        return len(get_shape_for_node_safely(model_input)) == 4


class GenericYoloTypeAnalyzer(GenericModelTypeAnalyzer):
    _topology_type = ModelType.YOLO

    @classmethod
    def is_like(cls, model_metadata: ModelMetaData) -> bool:
        return cls._has_region_yolo_node(model_metadata)

    @classmethod
    def _is_yolo_like(cls, model_metadata: ModelMetaData, output_count: int) -> bool:
        """
        Special check for models that dont have RegionYolo outputs.
        Criteria: single image input and outputs with proportional shapes
        (or single image output with odd number of cells).
        """
        if (len(model_metadata.outputs) != output_count or
                not cls._all_outputs_are_images(model_metadata)):
            return False
        return (
                cls._check_single_image_input(model_metadata) and
                cls._check_output_shape_proportions(model_metadata)
        )

    @staticmethod
    def _is_yolo_shape(shape: List[int]) -> bool:
        # Shape objects don't support slicing
        target_dims = [shape[1], shape[2], shape[3]]
        return all(dim % 2 and dim > 1 for dim in target_dims)

    @classmethod
    def _check_output_shape_proportions(cls, model_metadata: ModelMetaData) -> bool:
        """
        Check if all outputs have proportional shapes (shapes of one of outputs are others' common divisors).
        Example: [1, 255, 17, 17], [1, 255, 34, 34]
        For single output, check if it has an odd number of cells in a row/col (as it is common practice).
        """
        output_shapes = [get_shape_for_node_safely(output) for output in model_metadata.outputs]

        if len(output_shapes) == 1:
            output_shape = output_shapes[0]
            return cls._is_yolo_shape(output_shape)

        zipped_shapes = zip(*output_shapes)
        for shape in zipped_shapes:
            if any(dim % min(shape) for dim in shape):
                return False
        return True

    @staticmethod
    def _has_region_yolo_node(model_metadata: ModelMetaData) -> bool:
        return 'RegionYolo' in model_metadata.ops_types

    @staticmethod
    def _get_anchors(model_metadata: ModelMetaData) -> Optional[List[float]]:
        anchors = None

        region_yolo_op = next(filter(lambda op: op.get_type_name() == 'RegionYolo', model_metadata.ops))
        if region_yolo_op:
            region_yolo_attributes = region_yolo_op.get_attributes()
            anchors = region_yolo_attributes.get('anchors', [])

        return anchors


class YoloV2TypeAnalyzer(GenericYoloTypeAnalyzer):
    _topology_type = ModelType.YOLO_V2

    @classmethod
    def is_like(cls, model_metadata: ModelMetaData) -> bool:
        return (
                cls._has_region_yolo_node(model_metadata) and
                cls._get_anchors(model_metadata) == YoloAnchors.yolo_v2
        )


class TinyYoloV2TypeAnalyzer(GenericYoloTypeAnalyzer):
    _topology_type = ModelType.TINY_YOLO_V2

    @classmethod
    def is_like(cls, model_metadata: ModelMetaData) -> bool:
        if cls._has_region_yolo_node(model_metadata):
            return cls._get_anchors(model_metadata) == YoloAnchors.tiny_yolo_v2.value
        return cls._is_yolo_like(model_metadata, output_count=1)


class YoloV4TypeAnalyzer(GenericYoloTypeAnalyzer):
    _topology_type = ModelType.YOLO_V4

    @classmethod
    def is_like(cls, model_metadata: ModelMetaData) -> bool:
        return cls._is_yolo_like(model_metadata, output_count=3)


class YoloV3TypeAnalyzer(GenericYoloTypeAnalyzer):
    _topology_type = ModelType.YOLO_V3

    @classmethod
    def is_like(cls, model_metadata: ModelMetaData) -> bool:
        return (
                cls._has_region_yolo_node(model_metadata) and
                cls._get_anchors(model_metadata) == YoloAnchors.yolo_v3.value
        )


class TinyYoloV3V4TypeAnalyzer(GenericYoloTypeAnalyzer):
    _topology_type = ModelType.YOLO_V3

    @classmethod
    def is_like(cls, model_metadata: ModelMetaData) -> bool:
        if cls._has_region_yolo_node(model_metadata):
            return cls._get_anchors(model_metadata) == YoloAnchors.tiny_yolo_v3_v4.value

        return cls._is_yolo_like(model_metadata, output_count=2)


class SSDTypeAnalyzer(GenericModelTypeAnalyzer):
    _topology_type = ModelType.SSD

    @classmethod
    def is_like(cls, model_metadata: ModelMetaData) -> bool:
        op_types = model_metadata.ops_types
        return 'ROIPooling' not in op_types and 'DetectionOutput' in op_types


class ClassificationTypeAnalyzer(GenericModelTypeAnalyzer):
    @classmethod
    def is_like(cls, model_metadata: ModelMetaData) -> bool:
        if len(model_metadata.outputs) != 1:
            return False

        layer_types = model_metadata.ops_types
        excluded_types = {'PRelu', 'NormalizeL2'}
        valid_layer_types = not set(layer_types) & excluded_types

        if not valid_layer_types:
            return False

        output_layer = model_metadata.outputs[0]
        out_shape = get_shape_for_node_safely(output_layer)

        if len(out_shape) == 2:
            return True

        if len(out_shape) < 4:
            return False

        # To qualify, the outputs' HW shapes must either be missing or equal 1
        return out_shape[2] == out_shape[3] == 1 and out_shape[1] > 1


class InstanceSegmentationTypeAnalyzer(GenericModelTypeAnalyzer):
    _topology_type = ModelType.INSTANCE_SEGM

    @classmethod
    def is_like(cls, model_metadata: ModelMetaData) -> bool:
        layer_types = model_metadata.ops_types

        # ONNX Instance Segmentation has at least 2 ROIFeatureExtractor layers
        # TF Instance Segmentation is similar to Faster-RCNN, but with additional layers after detection
        return (
                ('ROIPooling' in layer_types and 'DetectionOutput' not in layer_types)
                or 'ExperimentalDetectronROIFeatureExtractor' in layer_types
        )


class SemanticSegmentationTypeAnalyzer(GenericModelTypeAnalyzer):
    _topology_type = ModelType.INSTANCE_SEGM

    @classmethod
    def is_like(cls, model_metadata: ModelMetaData) -> bool:
        if len(model_metadata.outputs) != 1 or len(model_metadata.inputs) > 1:
            return False

        ops_types = model_metadata.ops
        convolutions = filter(
            lambda op: op.get_type_name in ('Convolution', 'GroupConvolution'),
            ops_types
        )
        dilations = {str(layer.get_attributes()['dilations']) for layer in convolutions}
        dilations.discard(None)

        parameter = model_metadata.inputs[0]
        input_shape = get_shape_for_node_safely(parameter)
        input_layout = parse_node_layout(parameter.node)
        if 'H' not in input_layout or 'W' not in input_layout:
            return False

        input_dim = input_shape[input_layout.index('H')], input_shape[input_layout.index('W')]

        result = model_metadata.outputs[0]
        output_shape = get_shape_for_node_safely(result)
        output_layout = parse_node_layout(result.node)
        if 'H' not in output_layout or 'W' not in output_layout:
            return False

        output_dim = output_shape[output_layout.index('H')], output_shape[output_layout.index('W')]

        equal_dims = bool(input_dim == output_dim)

        return equal_dims and len(dilations) > 1 and 'Elu' not in ops_types


class InPaintingTypeAnalyzer(GenericModelTypeAnalyzer):
    _topology_type = ModelType.IN_PAINTING

    @classmethod
    def is_like(cls, model_metadata: ModelMetaData) -> bool:
        return 'Elu' in model_metadata.ops_types and len(model_metadata.inputs) == 2


class StyleTransferTypeAnalyzer(GenericModelTypeAnalyzer):
    _topology_type = ModelType.STYLE_TRANSFER

    @classmethod
    def is_like(cls, model_metadata: ModelMetaData) -> bool:
        return 'MVN' in model_metadata.ops_types


class SuperResolutionTypeAnalyzer(GenericModelTypeAnalyzer):
    _topology_type = ModelType.SUPER_RESOLUTION

    @classmethod
    def is_like(cls, model_metadata: ModelMetaData) -> bool:

        if not cls._all_outputs_are_images(model_metadata):
            return False

        single_stream = len(model_metadata.inputs) == 1 and len(model_metadata.outputs) == 1
        double_stream = len(model_metadata.inputs) == 2 and len(model_metadata.outputs) == 1

        input_shapes = [get_shape_for_node_safely(_input) for _input in model_metadata.inputs]

        output_shape = next(get_shape_for_node_safely(output) for output in model_metadata.outputs)

        # Super-resolution model should return a valid RGB/grayscale image
        # Check the number of color channels and output dimensions
        if output_shape[1] not in (1, 3) or len(output_shape) == 2 or output_shape[2] == output_shape[3] == 1:
            return False

        proportional_dims = False
        if single_stream:
            proportional_dims = input_shapes[0][2] / output_shape[2] == input_shapes[0][3] / output_shape[3]
        elif double_stream:
            for input_shape in input_shapes:
                if input_shape[2] != output_shape[2]:
                    proportional_dims = input_shape[2] / output_shape[2] == input_shape[3] / output_shape[3]

        return single_stream or double_stream and proportional_dims


class FaceRecognitionTypeAnalyzer(GenericModelTypeAnalyzer):
    _topology_type = ModelType.FACE_RECOGNITION

    @classmethod
    def is_like(cls, model_metadata: ModelMetaData) -> bool:
        """
        Check if given model is used for Face Recognition.
            Criteria:
                1) Uses PRelu activation functions or separate L2 regularization;
                2) Single output with NC shape.
        """
        if len(model_metadata.outputs) != 1:
            return False

        output_layer = model_metadata.outputs[0]
        output_shapes = get_shape_for_node_safely(output_layer)

        return {'PRelu', 'NormalizeL2'} & model_metadata.ops_types and len(output_shapes) == 2


class LandmarkDetectionTypeAnalyzer(GenericModelTypeAnalyzer):
    _topology_type = ModelType.LANDMARK_DETECTION

    @classmethod
    def is_like(cls, model_metadata: ModelMetaData) -> bool:
        """
        Check if given model is used for Landmark Detection.
            Criteria:
                1) Uses PRelu activation functions;
                2) Single output with NCHW shape, H and W shapes reduced to 1px.
        """

        reduced_dims = False
        if len(model_metadata.outputs) == 1:
            output_layer = model_metadata.outputs[0]
            output_shapes = get_shape_for_node_safely(output_layer)
            if len(output_shapes) < 4:
                return False
            reduced_dims = output_shapes[2] == output_shapes[3] == 1

        return 'PRelu' in model_metadata.ops_types and reduced_dims

