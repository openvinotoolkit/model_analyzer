# Copyright (C) 2019-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union, Set
from xml.etree import ElementTree

# pylint: disable=import-error
from openvino.runtime import ConstOutput, Node, Model, Layout

from model_analyzer.constants import LayoutTypes
from model_analyzer.layout_utils import is_batched_image_layout, parse_node_layout, is_image_info_layout
from model_analyzer.openvino_core_service import OPENVINO_CORE_SERVICE
from model_analyzer.shape_utils import get_shape_for_node_safely


# pylint: disable=too-many-public-methods
class ModelMetaData:
    """Retrieve IR metadata using heuristics."""

    def __init__(self, model_path: Path, weights_path: Path):
        self._model: Model = OPENVINO_CORE_SERVICE.read_model(str(model_path), str(weights_path))

        self._ops: List[Node] = self.model.get_ordered_ops()

        # Compile model to get execution graph if needed before Constant Folding (WA for PriorBox)
        self.int8precisions, self.int8layers = self.get_exec_graph_int8layers()

        OPENVINO_CORE_SERVICE.pass_constant_folding(self.model)

        self._model_file_suffix = model_path.suffix
        self.xml = None if self._is_onnx else ElementTree.parse(model_path)

    @property
    def model(self) -> Model:
        return self._model

    @property
    def ops(self) -> List[Node]:
        return self._ops

    @property
    def ops_types(self) -> Set[Node]:
        return {node.get_type_name() for node in self.ops}

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
        if not self.xml:
            return None

        ir_version = self.xml.getroot().attrib.get('version')
        try:
            return int(ir_version)
        except (TypeError, ValueError):
            return None

    def get_opsets(self) -> List[str]:
        op_sets = set()
        for op in self.ops:
            type_info = op.type_info
            op_sets.add(type_info.version_id)
        return sorted(list(op_sets))

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
        return [model_input.node.name for model_input in self.inputs]

    @property
    def output_names(self) -> List[str]:
        return [model_output.node.name for model_output in self.outputs]

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

        if 'RegionYolo' in self._layer_types:
            operation = next(filter(lambda operation: operation.get_type_name() == 'RegionYolo', self.ops))
            params = operation.get_attributes()
            num_classes = params['classes']
        elif 'DetectionOutput' in self._layer_types:
            operation = next(filter(lambda operation: operation.get_type_name() == 'DetectionOutput', self.ops))
            params = operation.get_attributes()
            num_classes = params['num_classes']
        elif 'SoftMax' in self._layer_types:
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
            shape = get_shape_for_node_safely(node)
            indicator = len(shape) == 2 and shape[1] == 1001
        return True if indicator else None

    def get_layers_ids(self) -> Dict[str, str]:
        layer_names = [x.name for x in self.ops]
        return {layer_names[i]: i for i in range(len(layer_names))}

    def get_exec_graph_int8layers(self, device: str = 'CPU') -> Tuple[list, list]:
        int8layers = []
        int8precisions = set()
        # pylint: disable=too-many-nested-blocks
        if not self.is_int8():
            return [], []
        compiled_model = OPENVINO_CORE_SERVICE.compile_model(self.model, device)
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
    def get_output_shape(output: ConstOutput) -> List[int]:
        return get_shape_for_node_safely(output)
