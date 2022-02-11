# Copyright (C) 2019-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union
from xml.etree import ElementTree

# pylint: disable=import-error
from openvino.runtime import ConstOutput, Node, Model, Layout
from openvino.runtime.passes import Manager

from model_analyzer.constants import ModelTypes, YoloAnchors, LayoutTypes
from model_analyzer.layout_utils import is_batched_image_layout, parse_node_layout, is_image_info_layout
from model_analyzer.openvino_core_service import OPENVINO_CORE_SERVICE
from model_analyzer.shape_utils import get_shape_for_node_safely


# pylint: disable=too-many-public-methods
class ModelMetaData:
    """Retrieve IR metadata using heuristics."""

    def __init__(self, model_path: Path, weights_path: Path):
        self.model: Model = OPENVINO_CORE_SERVICE.read_model(str(model_path), str(weights_path))

        self.ops: List[Node] = self.model.get_ordered_ops()

        # Compile model to get execution graph if needed before Constant Folding (WA for PriorBox)
        self.int8precisions, self.int8layers = self.get_exec_graph_int8layers()

        self._constant_folding()

        self._model_file_suffix = model_path.suffix
        self.xml = None if self._is_onnx else ElementTree.parse(model_path)

    def _constant_folding(self):
        try:
            pass_manager = Manager()
            pass_manager.register_pass('ConstantFolding')
            pass_manager.set_per_pass_validation(False)
            pass_manager.run_passes(self.model)
        # pylint: disable=broad-except
        except Exception as exception:
            logging.error(exception, exc_info=True)

    @property
    def _is_onnx(self) -> bool:
        return self._model_file_suffix in ('.onnx', '.prototxt')

    @property
    def batch(self) -> Optional[int]:
        for model_input in self.inputs:
            parameter_node = model_input.node
            parameter_layout: Layout = parameter_node.layout
            if parameter_layout.empty:
                continue

            if parameter_layout.has_name('N'):
                batch_index = parameter_layout.get_index_by_name('N')
                return get_shape_for_node_safely(parameter_node)[batch_index]
        return None

    def get_ir_version(self) -> Optional[int]:
        """Return IR version or `None` if the attribute is absent."""
        ir_version = self.xml.getroot().attrib.get('version')
        try:
            return int(ir_version)
        except (TypeError, ValueError):
            return None

    def get_opsets(self) -> list:
        opsets = set()
        layers = list(self.xml.getroot().find('layers'))
        for layer in layers:
            layer_version = layer.attrib.get('version').lower()
            if 'opset' in layer_version:
                opsets.add(layer_version)
        return sorted(list(opsets))

    def is_obsolete(self) -> bool:
        return not bool(self.model)

    def get_framework(self) -> str:
        if self._is_onnx:
            return 'onnx'
        return self.xml.find('./meta_data/cli_parameters/framework').attrib['value']

    @property
    def outputs(self) -> List[ConstOutput]:
        return self.model.outputs

    @property
    def inputs(self) -> List[ConstOutput]:
        return self.model.inputs

    @property
    def input_names(self) -> List[str]:
        return [model_input.any_name for model_input in self.inputs]

    @property
    def output_names(self) -> List[str]:
        return [model_input.any_name for model_input in self.outputs]

    @property
    def _layer_types(self) -> List[str]:
        return [layer.get_type_name() for layer in self.ops]

    def find_input_info_layer(self) -> Optional[str]:
        """Return the name of the IMAGE_INFO layer. Instance segmentation only."""
        for model_input in self.inputs:
            layout = parse_node_layout(model_input.node)
            if is_image_info_layout(layout):
                return model_input.any_name
        return None

    def analyze_inpainting_inputs(self) -> Dict[str, str]:
        """Return input predictions for image and mask layers. InPainting only."""
        roles = {}
        for model_input in self.inputs:
            parameter_node = model_input.node
            shape = get_shape_for_node_safely(parameter_node)
            layout = parse_node_layout(parameter_node)
            if not is_batched_image_layout(layout):
                continue
            c_index = layout.index('C')
            if shape[c_index] == 1:  # C dimension is 1
                roles['mask'] = model_input.any_name
            elif shape[c_index] == 3:  # C dimension is 3:
                roles['image'] = model_input.any_name

        return roles

    def analyze_super_resolution_inputs(self) -> Dict[str, str]:
        """Return input predictions for low-res and upsampling inputs. Super-resolution only."""
        roles = {}
        if len(self.input_names) == 1:
            roles['low-res'] = self.input_names[0]
            return roles

        dims = {}
        for candidate in self.inputs:
            shape = get_shape_for_node_safely(candidate)
            if shape[2]:
                dims[candidate.any_name] = shape[2]
        dims_sorted = [k for k, _ in sorted(dims.items(), key=lambda item: item[1])]
        roles['low-res'] = dims_sorted[0]
        roles['upsample'] = dims_sorted[1] if len(dims_sorted) > 1 else None

        return roles

    def analyze_output_roles(self) -> Optional[Dict[str, str]]:
        """Return role predictions for output layers. Instance Segmentation only."""
        roles = {}
        framework = self.get_framework()
        if framework == 'onnx':
            roles = self._get_output_roles_for_instance_segm_from_onnx()
        elif framework == 'tf':
            roles = self._get_output_roles_for_instance_segm_from_tf()
        return roles or None

    def _get_output_roles_for_instance_segm_from_onnx(self) -> Dict[str, str]:
        roles = {}
        for result in self.outputs:
            node = result.node
            result_precision = node.get_output_element_type(0).get_type_name()
            layout = parse_node_layout(node)

            if result_precision in {'i32', 'i16'}:
                roles['classes_out'] = result.any_name
                continue
            if result_precision in {'f32', 'f16'} and layout == LayoutTypes.C:
                roles['scores_out'] = result.any_name
                continue
            if layout == LayoutTypes.NC:
                roles['boxes_out'] = result.any_name
                continue
            if is_batched_image_layout(layout):  # Layout is NCHW
                roles['raw_masks_out'] = result.any_name
        return roles

    def _get_output_roles_for_instance_segm_from_tf(self) -> Dict[str, str]:
        roles = {}
        for result in self.outputs:
            node = result.node
            layout = parse_node_layout(node)
            if layout == LayoutTypes.NC:
                roles['detection_out'] = result.any_name
                continue
            if is_batched_image_layout(layout):
                roles['raw_masks_out'] = result.any_name
        return roles

    def get_yolo_v2_params(self) -> Dict[str, Union[str, int]]:
        """Extract model params from the output layer of the model. YOLOv2/TinyYOLOv2 only."""
        params = {}
        relevant_attributes = {'classes', 'coords', 'num'}
        output_attributes = self.outputs[0].node.get_attributes()
        for attribute in relevant_attributes:
            params[attribute] = output_attributes.get(attribute)

        return params

    def is_argmax_used(self):
        """Return info on whether the model output is argmaxed. Semantic Segmentation only"""
        output_node = self.outputs[0]
        layout = parse_node_layout(output_node)

        c_index = layout.index('C')
        if not c_index:
            return False

        output_shape = get_shape_for_node_safely(output_node)
        return output_shape[c_index] == 1

    def get_mo_params(self) -> Optional[Dict[str, str]]:
        """Return Model Optimizer CLI parameters from IR metadata, `None` if the node is absent."""

        mo_cli_params_node = self.xml.find('./meta_data/cli_parameters')
        mo_version_node = self.xml.find('./meta_data/MO_version')
        mo_cli_params = {}
        if mo_cli_params_node is not None:
            mo_cli_params = {n.tag: n.attrib['value'] for n in mo_cli_params_node if n.tag != 'unset'}
        if mo_version_node is not None:
            mo_cli_params['version'] = mo_version_node.attrib['value']
        return mo_cli_params or None

    def has_layer_of_type(self, *layer_types: str) -> bool:
        """Return True if the model has a layer whose type is in `layer_types`."""
        for layer in self.ops:
            if layer.get_type_name() in layer_types:
                return True
        return False

    def is_int8(self) -> bool:
        """Return True if the model was Int8 quantized."""
        return self.has_layer_of_type('FakeQuantize')

    def is_winograd(self) -> bool:
        """Return True if the model was adapted for Winograd algorithm."""

        for layer in self.ops:
            if layer.get_type_name() == 'Convolution' and 'PrimitivesPriority' in layer.rt_info and \
                    'cpu:jit_avx512_winograd' in layer.rt_info['PrimitivesPriority'].get():
                return True
        return False

    def get_num_classes(self) -> Optional[int]:
        """Return number of classes the IR supports, if possible."""
        if len(self.outputs) != 1:
            return None

        layer_types = self._layer_types

        if 'RegionYolo' in layer_types:
            operation = next(filter(lambda operation: operation.get_type_name() == 'RegionYolo', self.ops))
            params = operation.get_attributes()
            num_classes = params['classes']
        elif 'DetectionOutput' in layer_types:
            operation = next(filter(lambda operation: operation.get_type_name() == 'DetectionOutput', self.ops))
            params = operation.get_attributes()
            num_classes = params['num_classes']
        elif 'SoftMax' in layer_types:
            operation = next(filter(lambda operation: operation.get_type_name().lower() == 'SoftMax', self.ops))
            out_shape = self._get_output_shape(operation)
            num_classes = out_shape[1]
        else:
            return None

        return int(num_classes)

    def has_background_class(self) -> Optional[bool]:
        """Return True if the IR supports background class, None if unknown."""
        if len(self.outputs) != 1:
            return None

        output = self.outputs[0]

        node = output.node
        output_type = node.get_type_name().lower()
        params = node.get_attributes()

        indicator = False
        if output_type == 'regionyolo':
            indicator = 'background_label_id' in params
        elif output_type == 'detectionoutput':
            indicator = 'attrs.background_label_id' in params or 'background_label_id' in params
        elif output_type == 'softmax':
            shape = self._get_output_shape(output)
            indicator = len(shape) == 2 and shape[1] == 1001
        return True if indicator else None

    def _get_anchors(self) -> Optional[List[float]]:
        region_yolo = [output for output in self.outputs if output.node.get_type_name() == 'RegionYolo']
        if region_yolo:
            return region_yolo[0].get_attributes().get('anchors', [])
        return None

    def yolo_has_raw_output(self) -> bool:
        return 'RegionYolo' not in [output.node.get_type_name() for output in self.outputs]

    def _is_yolo(self) -> bool:
        layer_types = set(self._layer_types)
        return 'RegionYolo' in layer_types

    def _is_yolo_v2(self) -> bool:
        return self._is_yolo() and self._get_anchors() == YoloAnchors.yolo_v2.value

    def _is_tiny_yolo_v2(self) -> bool:
        return self._get_anchors() == YoloAnchors.tiny_yolo_v2.value if self._is_yolo() \
            else self._is_yolo_like(output_count=1)

    def _is_yolo_v3(self) -> bool:
        return self._is_yolo() and self._get_anchors() == YoloAnchors.yolo_v3.value

    def _is_yolo_v4(self) -> bool:
        return self._is_yolo_like(output_count=3)

    def _is_tiny_yolo_v3_v4(self) -> bool:
        """If Tiny YOLO v3 - check for anchors like YOLO v3, else treat like YOLO v4 with 2 outputs"""
        if self._is_yolo():
            return self._get_anchors() == YoloAnchors.tiny_yolo_v3_v4.value

        return self._is_yolo_like(output_count=2)

    def _is_yolo_like(self, output_count: int) -> bool:
        """
        Special check for models that dont have RegionYolo outputs.
        Criteria: single image input and outputs with proportional shapes
        (or single image output with odd number of cells).
        """
        if self._all_outputs_are_images() and len(self.outputs) == output_count:
            return self._check_single_image_input() and self._check_output_shape_proportions()

        return False

    def _check_single_image_input(self) -> bool:
        if not len(self.input_names) == 1:
            return False
        return len(get_shape_for_node_safely(self.model.input())) == 4

    def _all_outputs_are_images(self) -> bool:
        return all(len(self._get_output_shape(output)) == 4 for output in self.outputs)

    def _check_output_shape_proportions(self) -> bool:
        """
        Check if all outputs have proportional shapes (shapes of one of outputs are others' common divisors).
        Example: [1, 255, 17, 17], [1, 255, 34, 34]
        For single output, check if it has an odd number of cells in a row/col (as it is common practice).
        """
        output_shapes = [self._get_output_shape(output) for output in self.outputs]

        if len(output_shapes) == 1:
            return ModelMetaData._is_yolo_shape(output_shapes[0])

        zipped_shapes = zip(*output_shapes)
        for shape in zipped_shapes:
            if any(dim % min(shape) for dim in shape):
                return False
        return True

    @staticmethod
    def _is_yolo_shape(shape) -> bool:
        # Shape objects don't support slicing
        target_dims = [shape[1], shape[2], shape[3]]
        odd_dims = all(dim % 2 for dim in target_dims)
        not_reduced = all(dim > 1 for dim in target_dims)

        return odd_dims and not_reduced

    def guess_topology_type(self) -> ModelTypes:
        """Return type of the topology or 'generic' if unknown."""

        topology_types = {
            ModelTypes.YOLO_V2: self._is_yolo_v2,
            ModelTypes.TINY_YOLO_V2: self._is_tiny_yolo_v2,
            ModelTypes.YOLO_V3: self._is_yolo_v3,
            ModelTypes.YOLO_V4: self._is_yolo_v4,
            ModelTypes.TINY_YOLO_V3_V4: self._is_tiny_yolo_v3_v4,
            ModelTypes.YOLO: self._is_yolo,
            ModelTypes.SSD: self._is_ssd,
            ModelTypes.CLASSIFICATION: self._is_classification,
            ModelTypes.INSTANCE_SEGM: self._is_instance_segmentation,
            ModelTypes.SEMANTIC_SEGM: self._is_semantic_segmentation,
            ModelTypes.INAPINTING: self._is_inpainting,
            ModelTypes.STYLE_TRANSFER: self._is_style_transfer,
            ModelTypes.SUPER_RESOLUTION: self._is_super_resolution,
            ModelTypes.FACE_RECOGNITION: self._is_face_recognition,
            ModelTypes.LANDMARK_DETECTION: self._is_landmark_detection
        }

        for t_type, t_detector in topology_types.items():
            if t_detector():
                return t_type
        return ModelTypes.GENERIC

    def _is_classification(self) -> bool:
        if len(self.outputs) != 1:
            return False

        layer_types = set(self._layer_types)
        excluded_types = {'PRelu', 'NormalizeL2'}
        valid_layer_types = not layer_types & excluded_types

        if not valid_layer_types:
            return False

        out_layer = self.outputs[0]
        out_shape = self._get_output_shape(out_layer)

        if len(out_shape) == 2:
            return True

        if len(out_shape) < 4:
            return False
        reduced_shapes = out_shape[2] == out_shape[3] == 1 and out_shape[1] > 1

        # To qualify, the outputs' HW shapes must either be missing or equal 1
        return reduced_shapes and valid_layer_types

    def _is_ssd(self) -> bool:
        layer_types = set(self._layer_types)
        return 'ROIPooling' not in layer_types and 'DetectionOutput' in layer_types

    def _is_instance_segmentation(self) -> bool:
        if not self.xml:
            return False

        layer_types = set(self._layer_types)
        output_types = {output.node.get_type_name() for output in self.outputs}

        xml_layers = list(self.xml.getroot().find('layers'))
        xml_layer_types = [layer.attrib.get('type') for layer in xml_layers]

        # ONNX Instance Segmentation has at least 2 ROIFeatureExtractor layers
        # TF Instance Segmentation is similar to Faster-RCNN, but with additional layers after detection
        return (
                ('ROIPooling' in layer_types and 'DetectionOutput' not in output_types)
                or 'ExperimentalDetectronROIFeatureExtractor' in xml_layer_types
        )

    def _is_semantic_segmentation(self) -> bool:
        if len(self.outputs) != 1:
            return False

        convolutions = (layer for layer in self.ops if layer.get_type_name() in ('Convolution', 'GroupConvolution'))
        dilations = {str(layer.get_attributes()['dilations']) for layer in convolutions}
        dilations.discard(None)

        layers_types = self._layer_types

        if len(self.inputs) > 1:
            return False

        input_layer = self.inputs[0]
        input_shape = get_shape_for_node_safely(input_layer)
        input_layout = parse_node_layout(input_layer.node)
        if 'H' not in input_layout or 'W' not in input_layout:
            return False

        input_dim = input_shape[input_layout.index('H')], input_shape[input_layout.index('W')]

        output_layer = self.outputs[0]
        output_shape = self._get_output_shape(output_layer)
        output_layout = parse_node_layout(output_layer.node)
        if 'H' not in output_layout or 'W' not in output_layout:
            return False

        output_dim = output_shape[output_layout.index('H')], output_shape[output_layout.index('W')]

        equal_dims = bool(input_dim == output_dim)

        return equal_dims and len(dilations) > 1 and 'Elu' not in layers_types

    def _is_inpainting(self) -> bool:
        layers_types = set(self._layer_types)
        inputs = self.input_names

        return 'Elu' in layers_types and len(inputs) == 2

    def _is_style_transfer(self) -> bool:
        layers_types = set(self._layer_types)

        return 'MVN' in layers_types

    def _is_super_resolution(self) -> bool:

        if not self._all_outputs_are_images():
            return False

        single_stream = len(self.input_names) == 1 and len(self.outputs) == 1
        double_stream = len(self.input_names) == 2 and len(self.outputs) == 1

        input_nodes = [model_input.node for model_input in self.model.inputs]
        input_shapes = [get_shape_for_node_safely(candidate) for candidate in input_nodes]

        output_shape = next(self._get_output_shape(candidate) for candidate in self.outputs)

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

    def _is_face_recognition(self) -> bool:
        """
        Check if given model is used for Face Recognition.
            Criteria:
                1) Uses PRelu activation functions or separate L2 regularization;
                2) Single output with NC shape.
        """
        if len(self.outputs) != 1:
            return False

        output_layer = self.outputs[0]
        output_shapes = self._get_output_shape(output_layer)
        layers_types = set(self._layer_types)

        return {'PRelu', 'NormalizeL2'} & layers_types and len(output_shapes) == 2

    def _is_landmark_detection(self) -> bool:
        """
        Check if given model is used for Landmark Detection.
            Criteria:
                1) Uses PRelu activation functions;
                2) Single output with NCHW shape, H and W shapes reduced to 1px.
        """
        layers_types = set(self._layer_types)

        reduced_dims = False
        if len(self.outputs) == 1:
            output_layer = self.outputs[0]
            output_shapes = self._get_output_shape(output_layer)
            if len(output_shapes) < 4:
                return False
            reduced_dims = output_shapes[2] == output_shapes[3] == 1

        return 'PRelu' in layers_types and reduced_dims

    def get_layers_ids(self) -> Dict[str, str]:
        layer_names = [x.name for x in self.ops]
        return {layer_names[i]: i for i in range(len(layer_names))}

    def get_exec_graph_int8layers(self) -> Tuple[list, list]:
        int8layers = []
        int8precisions = set()
        # pylint: disable=too-many-nested-blocks
        if not self.is_int8():
            return [], []
        compiled_model = OPENVINO_CORE_SERVICE.compile_model(self.model, 'CPU')
        runtime_model = compiled_model.get_runtime_model()
        for execution_node in runtime_model.get_ordered_ops():
            rt_info = execution_node.get_rt_info()
            layer_type = rt_info['layerType']
            inputs_number = (
                1 if layer_type.lower() in {'convolution', 'deconvolution', 'fullyconnected', 'gemm', 'pooling'}
                else len(execution_node.inputs())
            )
            input_precisions = [
                execution_node.input(i).get_source_output().node.get_rt_info()['outputPrecisions'].lower()
                for i in range(inputs_number)]
            search_precisions = ['i8', 'u8']
            for precision in search_precisions:
                if precision in input_precisions:
                    int8precisions.add(precision)
            is_int8 = all(p in search_precisions for p in input_precisions) and input_precisions
            if is_int8:
                original_layers_names = rt_info['originalLayersNames']
                if original_layers_names:
                    original_layers_names = original_layers_names.split(',')
                    int8layers += original_layers_names

        return list(int8precisions), int8layers

    def is_model_dynamic(self) -> bool:
        return self.model.is_dynamic()

    @staticmethod
    def _get_output_shape(output: ConstOutput) -> List[int]:
        return get_shape_for_node_safely(output)
