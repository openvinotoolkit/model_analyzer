# Copyright (C) 2019-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Dict, Optional, Tuple, List, Set
from xml.etree import ElementTree

# pylint: disable=import-error
from openvino.runtime import ConstOutput, Node, Model, Layout

from model_analyzer.layout_utils import parse_node_layout, is_image_info_layout
from model_analyzer.openvino_core_service import OPENVINO_CORE_SERVICE
from model_analyzer.precision_service import Precision, PRECISION_SERVICE
from model_analyzer.shape_utils import get_shape_for_node_safely


# pylint: disable=too-many-public-methods
from tests.constants import MODELS_PATH


class ModelMetaData:
    """Retrieve IR metadata using heuristics."""

    def __init__(self, model_path: Path, weights_path: Path):
        self._model_path = model_path.name
        self._model: Model = OPENVINO_CORE_SERVICE.read_model(str(model_path), str(weights_path))

        self._ops: List[Node] = self.model.get_ordered_ops()

        # Compile model to get execution graph if needed before Constant Folding (WA for PriorBox)
        self._precisions_distributions = self._get_ops_by_exec_precisions()

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
    def execution_precisions(self) -> Tuple[Precision]:
        return tuple(i.value for i in self._precisions_distributions)

    def get_execution_precisions(self, layer_name: str) -> Precision:
        for precision, layer_names in self._precisions_distributions.items():
            if layer_name in layer_names:
                return precision
        return Precision.unknown

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

    @property
    def ir_version(self) -> Optional[int]:
        """Return IR version or `None` if the attribute is absent."""
        if not self.xml:
            return None

        ir_version = self.xml.getroot().attrib.get('version')
        try:
            return int(ir_version)
        except (TypeError, ValueError):
            return None

    @property
    def op_sets(self) -> List[str]:
        op_sets = set()
        for operation in self.ops:
            type_info = operation.type_info
            op_sets.add(type_info.version_id)
        return sorted(list(op_sets))

    def is_obsolete(self) -> bool:
        return not bool(self.model)

    @property
    def framework(self) -> Optional[str]:
        if self._is_onnx:
            return 'onnx'
        if not self.xml:
            return None
        framework = self.xml.find('./meta_data/cli_parameters/framework')
        if not framework:
            return None
        return framework.attrib['value']

    @property
    def outputs(self) -> List[ConstOutput]:
        return self.model.outputs

    @property
    def inputs(self) -> List[ConstOutput]:
        return self.model.inputs

    def find_input_info_layer(self) -> Optional[str]:
        """Return the name of the IMAGE_INFO layer. Instance segmentation only."""
        for model_input in self.inputs:
            layout = parse_node_layout(model_input.node)
            if is_image_info_layout(layout):
                return model_input.any_name
        return None

    @property
    def mo_params(self) -> Optional[Dict[str, str]]:
        """Return Model Optimizer CLI parameters from IR metadata, `None` if the node is absent."""

        mo_cli_params_node = self.xml.find('./meta_data/cli_parameters')
        mo_version_node = self.xml.find('./meta_data/MO_version')
        mo_cli_params = {}
        if mo_cli_params_node is not None:
            mo_cli_params = {n.tag: n.attrib['value'] for n in mo_cli_params_node if n.tag != 'unset'}
        if mo_version_node is not None:
            mo_cli_params['version'] = mo_version_node.attrib['value']
        return mo_cli_params or None

    def has_op_of_type(self, *layer_types: str) -> bool:
        """Return True if the model has a layer whose type is in `layer_types`."""
        for layer in self.ops:
            if layer.get_type_name() in layer_types:
                return True
        return False

    def is_int8(self) -> bool:
        """Return True if the model was Int8 quantized."""
        return self.has_op_of_type('FakeQuantize')

    def is_winograd(self) -> bool:
        """Return True if the model was adapted for Winograd algorithm."""

        for layer in self.ops:
            if layer.get_type_name() == 'Convolution' and 'PrimitivesPriority' in layer.rt_info and \
                    'cpu:jit_avx512_winograd' in layer.rt_info['PrimitivesPriority'].get():
                return True
        return False

    @property
    def num_classes(self) -> Optional[int]:
        """Return number of classes the IR supports, if possible."""
        if len(self.outputs) != 1:
            return None

        if 'RegionYolo' in self.ops_types:
            operation = next(filter(lambda operation: operation.get_type_name() == 'RegionYolo', self.ops))
            params = operation.get_attributes()
            num_classes = params.get('classes')
        elif 'DetectionOutput' in self.ops_types:
            operation = next(filter(lambda operation: operation.get_type_name() == 'DetectionOutput', self.ops))
            params = operation.get_attributes()
            num_classes = params.get('num_classes')
        elif 'SoftMax' in self.ops_types:
            operation = next(filter(lambda operation: operation.get_type_name().lower() == 'SoftMax', self.ops))
            out_shape = get_shape_for_node_safely(operation)
            num_classes = out_shape[1]
        else:
            return None

        return int(num_classes) if num_classes else None

    @property
    def has_background_class(self) -> Optional[bool]:
        """Return True if the IR supports background class, None if unknown."""
        if len(self.outputs) != 1:
            return None

        output = self.outputs[0]

        node = output.node
        output_type = node.get_type_name()
        params = node.get_attributes()

        indicator = False
        if output_type == 'RegionYolo':
            indicator = 'background_label_id' in params
        elif output_type == 'DetectionOutput':
            indicator = 'attrs.background_label_id' in params or 'background_label_id' in params
        elif output_type == 'SoftMax':
            shape = get_shape_for_node_safely(node)
            indicator = len(shape) == 2 and shape[1] == 1001
        return True if indicator else None

    @property
    def ops_ids(self) -> Dict[str, int]:
        return {op.friendly_name: i for i, op in enumerate(self.ops)}

    def _get_ops_by_exec_precisions(self, device: str = 'CPU') -> Dict[Precision, List[str]]:
        precisions = {}

        compiled_model = OPENVINO_CORE_SERVICE.compile_model(self.model, device)
        runtime_model = compiled_model.get_runtime_model()
        path = MODELS_PATH / f'{self._model_path}'
        OPENVINO_CORE_SERVICE.serialize_model(runtime_model, f'{path}.xml', f'{path}.bin')
        for execution_node in runtime_model.get_ordered_ops():
            rt_info = execution_node.get_rt_info()

            raw_runtime_precision = rt_info['runtimePrecision']
            runtime_precision = PRECISION_SERVICE.get_precision(raw_runtime_precision)

            if runtime_precision not in precisions:
                precisions[runtime_precision] = []

            original_layers_names = rt_info['originalLayersNames']
            if not original_layers_names:
                continue
            precisions[runtime_precision].extend(original_layers_names.split(','))

        return precisions

    def get_execution_precision_for_op(self, layer_name: str) -> Optional[str]:
        for precision, layers in self._precisions_distributions.items():
            if layer_name in layers:
                return precision
        return None

    def is_model_dynamic(self) -> bool:
        return self.model.is_dynamic()
