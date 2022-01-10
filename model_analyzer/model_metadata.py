"""
 Model Analyzer

 Copyright (c) 2019 Intel Corporation

 LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”) is subject to
 the terms and conditions of the software license agreements for Software Package, which may also include
 notices, disclaimers, or license terms for third party or open source software
 included in or with the Software Package, and your use indicates your acceptance of all such terms.
 Please refer to the “third-party-programs.txt” or other similarly-named text file included with the Software Package
 for additional details.
 You may obtain a copy of the License at
      https://software.intel.com/content/dam/develop/external/us/en/documents/intel-openvino-license-agreements.pdf
"""
import logging
from contextlib import suppress
from pathlib import Path
from typing import Union, Dict, Optional, Tuple, List
from xml.etree import ElementTree

from openvino.runtime import Node, Model
from openvino.runtime.passes import Manager

from model_analyzer.constants import ModelTypes, YoloAnchors, LayoutTypes
from model_analyzer.openvino_core_service import OPENVINO_CORE_SERVICE
from model_analyzer.shape_utils import get_shape_safely, get_shape_for_node_safely


class ModelMetaData:
    """Retrieve IR metadata using heuristics."""

    def __init__(self, model_path: Path, weights_path: Path):
        self.model: Model = OPENVINO_CORE_SERVICE.read_model(str(model_path), str(weights_path))

        self.ops = self.model.get_ordered_ops()

        # Load network to get execution graph if needed before Constant Folding (WA for PriorBox)
        self.int8precisions, self.int8layers = self.get_exec_graph_int8layers()

        self.constant_folding()

        self.is_onnx = model_path.suffix in ('.onnx', '.prototxt')
        self.xml = None if self.is_onnx else ElementTree.parse(model_path)

        self.output_layers = [output.get_node() for output in self.model.outputs]
        self.input_layers = [model_input.node for model_input in self.model.inputs]

    def constant_folding(self):
        try:
            pass_manager = Manager()
            pass_manager.register_pass('ConstantFolding')
            pass_manager.set_per_pass_validation(False)
            pass_manager.run_passes(self.model)
        # pylint: disable=broad-except
        except Exception as exception:
            logging.error(exception, exc_info=True)

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

    def get_framework(self):
        return self.xml.find('./meta_data/cli_parameters/framework').attrib['value']

    def get_ie_outputs(self) -> List[str]:
        return [model_outputs.node for model_outputs in self.model.outputs]

    def get_ie_inputs(self) -> list:
        return [model_input.node.name for model_input in self.model.inputs]

    @staticmethod
    def _get_output_shape(layer: Node, port: int = 0) -> List[int]:
        return get_shape_for_node_safely(layer.output(port))

    def get_layer_types(self) -> list:
        return [layer.get_type_name() for layer in self.ops]

    def find_input_info_layer(self) -> str:
        """Return the name of the IMAGE_INFO layer. Instance segmentation only."""
        result = None
        for layer in self.get_ie_inputs():
            if self.model.input_info[layer].layout == LayoutTypes.NC.value:
                result = layer
                break
        return result

    def analyze_inpainting_inputs(self) -> Dict[str, str]:
        """Return input predictions for image and mask layers. Inpainting only."""
        roles = {}
        inputs = self.get_ie_inputs()
        for candidate in inputs:
            layer = self.network.input_info[candidate]
            shape = self.get_shape_values(layer.layout, layer.input_data.shape)
            if layer.layout == LayoutTypes.NCHW.value and shape['C'] == 1:
                roles['mask'] = candidate
            else:
                roles['image'] = candidate

        return roles

    def analyze_super_resolution_inputs(self) -> Dict[str, str]:
        """Return input predictions for low-res and upsampling inputs. Super-resolution only."""
        roles = {}
        if len(self.input_layers) == 1:
            roles['low-res'] = self.input_layers[0].get_friendly_name()
            return roles

        dims = {}
        for candidate in self.input_layers:
            shape = candidate.get_partial_shape().to_shape()
            if shape[2]:
                dims[candidate.get_friendly_name()] = shape[2]
        dims_sorted = [k for k, _ in sorted(dims.items(), key=lambda item: item[1])]
        roles['low-res'] = dims_sorted[0]
        roles['upsample'] = dims_sorted[1] if len(dims_sorted) > 1 else None

        return roles

    def analyze_output_roles(self) -> Union[Dict[str, str], None]:
        """Return role predictions for output layers. Instance Segmentation only."""
        roles = {}
        outputs = self.get_ie_outputs()
        framework = self.get_framework()
        if framework == 'onnx':
            for candidate in outputs:
                precision = candidate.get_output_element_type(0).get_type_name()
                if precision in {'i32', 'i16'}:
                    roles['classes_out'] = candidate
                    continue
                elif precision in {'fP32', 'fP16'} and self.network.outputs[candidate].layout == LayoutTypes.C.value:
                    roles['scores_out'] = candidate
                    continue
                if self.network.outputs[candidate].layout == LayoutTypes.NC.value:
                    roles['boxes_out'] = candidate
                    continue
                if self.network.outputs[candidate].layout == LayoutTypes.NCHW.value:
                    roles['raw_masks_out'] = candidate
        if framework == 'tf':
            for candidate in outputs:
                if self.network.outputs[candidate].layout == LayoutTypes.NC.value:
                    roles['detection_out'] = candidate
                    continue
                if self.network.outputs[candidate].layout == LayoutTypes.NCHW.value:
                    roles['raw_masks_out'] = candidate

        return roles if roles else None

    def get_yolo_v2_params(self) -> dict:
        """Extract model params from the output layer of the model. YOLOv2/TinyYOLOv2 only."""
        params = {}
        relevant_attributes = ['classes', 'coords', 'num']
        output_attributes = self.output_layers[0].get_attributes()
        for attribute in relevant_attributes:
            params[attribute] = output_attributes.get(attribute)

        return params

    def is_argmax_used(self):
        """Return info on whether the network output is argmaxed. Semantic Segmentation only"""
        output_layer = self.get_ie_outputs()[0]
        output_shape = self.get_shape_values(output_layer.layout, output_layer.shape)

        return output_shape['C'] == 1

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

        if self.model:
            for layer in self.ops:
                if layer.get_type_name() == 'Convolution' and 'PrimitivesPriority' in layer.rt_info and \
                        'cpu:jit_avx512_winograd' in layer.rt_info['PrimitivesPriority'].get():
                    return True
        else:
            for layer in self.network.layers.values():
                if layer.type == 'Convolution' and 'cpu:jit_avx512_winograd' in \
                        layer.params.get('PrimitivesPriority', ''):
                    return True

        return False

    def get_num_classes(self) -> Optional[int]:
        """Return number of classes the IR supports, if possible."""
        if len(self.output_layers) != 1:
            return None

        layer_types = self.get_layer_types()

        if 'RegionYolo' in layer_types:
            op = next(filter(lambda op: op.get_type_name() == 'RegionYolo', self.ops))
            params = op.get_attributes()
            num_classes = params['classes']
        elif 'DetectionOutput' in layer_types:
            op = next(filter(lambda op: op.get_type_name() == 'DetectionOutput', self.ops))
            params = op.get_attributes()
            num_classes = params['num_classes']
        elif 'SoftMax' in layer_types:
            op = next(filter(lambda op: op.get_type_name().lower() == 'softmax', self.ops))
            out_shape = self._get_output_shape(op)
            num_classes = out_shape[1]
        else:
            return None

        return int(num_classes)

    def has_background_class(self) -> Optional[bool]:
        """Return True if the IR supports background class, None if unknown."""
        if len(self.output_layers) != 1:
            return None

        output = self.output_layers[0]

        if isinstance(output, Node):
            output_type = output.get_type_name().lower()
            params = output.get_attributes()
        else:
            output_type = output.type.lower()
            params = output.params

        indicator = False
        if output_type == 'regionyolo':
            indicator = 'background_label_id' in params
        elif output_type == 'detectionoutput':
            indicator = 'attrs.background_label_id' in params or 'background_label_id' in params
        elif output_type == 'softmax':
            if isinstance(output, Node):
                shape = self._get_output_shape(output)
            else:
                shape = output.out_data[0].shape
            indicator = len(shape) == 2 and shape[1] == 1001
        return True if indicator else None

    @staticmethod
    def get_shape_values(layouts: list, shapes: list) -> dict:
        shape_values = {
            'N': None,
            'C': None,
            'H': None,
            'W': None
        }
        for dim in shape_values:
            if dim in layouts:
                shape_values[dim] = shapes[layouts.index(dim)]
        return shape_values

    def _get_anchors(self) -> Optional[List[float]]:
        region_yolo = [layer for layer in self.output_layers if layer.get_type_name() == 'RegionYolo']
        if region_yolo:
            return region_yolo[0].get_attributes().get('anchors', [])
        return None

    def yolo_has_raw_output(self) -> bool:
        return 'RegionYolo' not in [layer.get_type_name() for layer in self.output_layers if layer.get_type_name()]

    def _is_yolo(self) -> bool:
        layer_types = set(self.get_layer_types())
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
        Special check for networks that dont have RegionYolo outputs.
        Criteria: single image input and outputs with proportional shapes
        (or single image output with odd number of cells).
        """
        if self._all_outputs_are_image() and len(self.output_layers) == output_count:
            return self._check_single_image_input() and self._check_output_shape_proportions()

        return False

    def _check_single_image_input(self) -> bool:
        if not len(self.input_layers) == 1:
            return False
        return len(get_shape_for_node_safely(self.input_layers[0].output(0))) == 4

    def _all_outputs_are_image(self) -> bool:
        return all(len(self._get_output_shape(output)) == 4 for output in self.output_layers)

    def _check_output_shape_proportions(self) -> bool:
        """
        Check if all outputs have proportional shapes (shapes of one of outputs are others' common divisors).
        Example: [1, 255, 17, 17], [1, 255, 34, 34]
        For single output, check if it has an odd number of cells in a row/col (as it is common practice).
        """
        output_shapes = [self._get_output_shape(output) for output in self.output_layers]

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
        if len(self.output_layers) != 1:
            return False

        layer_types = set(self.get_layer_types())
        excluded_types = {'PRelu', 'NormalizeL2'}
        valid_layer_types = not layer_types & excluded_types

        out_layer = self.output_layers[0]
        out_shape = self._get_output_shape(out_layer)

        minimal_shape = len(out_shape) == 2
        if minimal_shape and valid_layer_types:
            return True
        reduced_shapes = out_shape[2] == out_shape[3] == 1 and out_shape[1] > 1

        # To qualify, the outputs' HW shapes must either be missing or equal 1
        return reduced_shapes and valid_layer_types

    def _is_ssd(self) -> bool:
        layer_types = set(self.get_layer_types())
        output_types = {layer.get_type_name() for layer in self.output_layers}

        return 'ROIPooling' not in layer_types and 'DetectionOutput' in output_types

    def _is_instance_segmentation(self) -> bool:
        if not self.xml:
            return False

        layer_types = set(self.get_layer_types())
        output_types = {layer.get_type_name() for layer in self.output_layers}

        xml_layers = list(self.xml.getroot().find('layers'))
        xml_layer_types = [layer.attrib.get('type') for layer in xml_layers]

        # ONNX Instance Segmentation has at least 2 ROIFeatureExtractor layers
        # TF Instance Segmentation is similar to Faster-RCNN, but with additional layers after detection
        return ('ROIPooling' in layer_types and 'DetectionOutput' not in output_types) \
               or 'ExperimentalDetectronROIFeatureExtractor' in xml_layer_types

    def _is_semantic_segmentation(self) -> bool:
        if len(self.output_layers) != 1:
            return False

        convolutions = [layer for layer in self.ops if layer.get_type_name() in ['Convolution', 'GroupConvolution']]
        dilations = {str(layer.get_attributes()['dilations']) for layer in convolutions}
        dilations.discard(None)

        layers_types = self.get_layer_types()

        input_layer = next(iter(self.model.inputs))

        input_shape = self.get_shape_values(input_layer.layout, input_layer.input_data.shape)

        output_layer = self.output_layers[0]
        output_shape = self._get_output_shape(output_layer)
        output_dim = list(output_shape)[-2:]

        input_dim = [input_shape['H'], input_shape['W']]
        equal_dims = bool(input_dim == output_dim)

        return equal_dims and len(dilations) > 1 and 'Elu' not in layers_types

    def _is_inpainting(self) -> bool:
        layers_types = set(self.get_layer_types())
        inputs = self.get_ie_inputs()

        return 'Elu' in layers_types and len(inputs) == 2

    def _is_style_transfer(self) -> bool:
        layers_types = set(self.get_layer_types())

        return 'MVN' in layers_types

    def _is_super_resolution(self) -> bool:
        single_stream = len(self.input_layers) == 1 and len(self.output_layers) == 1
        double_stream = len(self.input_layers) == 2 and len(self.output_layers) == 1

        input_shapes = [get_shape_for_node_safely(candidate) for candidate in self.input_layers]
        output_shape = [self._get_output_shape(candidate)
                        for candidate in self.output_layers][0]

        # Super-resolution network should return a valid RGB/grayscale image
        # Check the number of color channels and output dimensions
        if output_shape[1] not in (1, 3) or len(output_shape) == 2 or output_shape[2] == output_shape[3] == 1:
            return False

        proportional_dims = False
        if single_stream:
            proportional_dims = input_shapes[0][2] / output_shape[2] == \
                                input_shapes[0][3] / output_shape[3]
        elif double_stream:
            for input_shape in input_shapes:
                if input_shape[2] != output_shape[2]:
                    proportional_dims = input_shape[2] / output_shape[2] == \
                                        input_shape[3] / output_shape[3]

        return single_stream or double_stream and proportional_dims

    def _is_face_recognition(self) -> bool:
        """
        Check if given model is used for Face Recognition.
            Criteria:
                1) Uses PRelu activation functions or separate L2 regularization;
                2) Single output with NC shape.
        """
        if len(self.output_layers) != 1:
            return False

        output_layer = self.output_layers[0]
        output_shapes = self._get_output_shape(output_layer)
        layers_types = set(self.get_layer_types())

        return {'PRelu', 'NormalizeL2'} & layers_types and len(output_shapes) == 2

    def _is_landmark_detection(self) -> bool:
        """
        Check if given model is used for Landmark Detection.
            Criteria:
                1) Uses PRelu activation functions;
                2) Single output with NCHW shape, H and W shapes reduced to 1px.
        """
        layers_types = set(self.get_layer_types())

        reduced_dims = False
        if len(self.output_layers) == 1:
            output_layer = self.output_layers[0]
            output_shapes = self._get_output_shape(output_layer)
            reduced_dims = output_shapes[2] == output_shapes[3] == 1

        return 'PRelu' in layers_types and reduced_dims

    def get_layers_ids(self) -> Dict[str, str]:
        layer_names = [x.name for x in self.ops]
        return {layer_names[i]: i for i in range(len(layer_names))}

    def get_exec_graph_int8layers(self) -> Tuple[list, list]:
        int8layers = []
        int8precisions = set()
        # pylint: disable=too-many-nested-blocks
        if self.is_int8():
            compiled_model = OPENVINO_CORE_SERVICE.compile_model(self.model, 'CPU')
            layers_with_one_inputs = {
                'convolution', 'deconvolution',
                'fullyconnected', 'gemm', 'pooling'
            }
            try:
                execution_model = compiled_model.get_exec_graph_info()
                for layer in execution_model.get_ordered_ops():
                    rt_info = layer.get_rt_info()
                    layer_type = rt_info['layerType'].get()
                    inputs_number = 1 if layer_type.lower() in layers_with_one_inputs else len(layer.inputs())
                    input_precisions = [
                        layer.input(i).get_source_output().get_node().get_rt_info()['outputPrecisions'].get().lower()
                        for i in range(inputs_number)]
                    search_precisions = {'i8', 'u8'}
                    for precision in search_precisions:
                        if precision in input_precisions:
                            int8precisions.add(precision)
                    is_int8 = all(p in search_precisions for p in input_precisions) and input_precisions
                    if is_int8:
                        original_layers_names = rt_info['originalLayersNames'].get()
                        if original_layers_names:
                            original_layers_names = original_layers_names.split(',')
                            int8layers += original_layers_names
            finally:  # Avoiding IE Python API crash with SIGSEGV.
                with suppress(NameError):
                    del rt_info
                with suppress(NameError):
                    # pylint: disable=W0631
                    del layer
                del execution_model
        return list(int8precisions), int8layers

    def is_model_dynamic(self) -> bool:
        return self.model.is_dynamic()