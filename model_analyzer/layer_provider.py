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
import operator
import struct
from functools import reduce
from typing import List, Type, Tuple, Dict, Iterable

import numpy as np

# pylint: disable=import-error
from openvino.runtime import Node

from model_analyzer.shape_utils import get_shape_for_node_safely


def _get_param_values(params: dict, multiple_form: str, single_form: str) -> list:
    if multiple_form in params:
        if isinstance(params[multiple_form], list):
            value_per_dimension = params[multiple_form]
        else:
            value_per_dimension = [int(i) for i in params[multiple_form].split(',')]
    else:
        x_key = '{}-x'.format(single_form)
        y_key = '{}-y'.format(single_form)
        z_key = '{}-z'.format(single_form)
        x_dim = int(params.get(x_key, 1))
        y_dim = int(params.get(y_key, 1))
        z_dim = int(params.get(z_key, 1))
        value_per_dimension = [x_dim, y_dim, z_dim]
    return value_per_dimension


class LayerTypesManager:
    types_map = dict()

    @staticmethod
    def register_type_of_layer(new_class: Type['LayerType']) -> Type['LayerType']:
        for layer_type in [x.lower() for x in new_class.layer_types]:
            LayerTypesManager.types_map[layer_type] = new_class
        return new_class

    @staticmethod
    def provider(layer: Node) -> 'LayerType':
        layer_type = layer.get_type_name().lower()
        return LayerTypesManager.types_map.get(layer_type, LayerType)(layer)


class MetaClass(type):
    def __new__(cls, *args, **kwargs):
        new_class = super(MetaClass, cls).__new__(cls, *args, **kwargs)
        LayerTypesManager.register_type_of_layer(new_class)
        return new_class


# pylint: disable=R0904
class LayerType(metaclass=MetaClass):
    layer_types = ['GeneralIE']

    def __init__(self, layer: Node):
        self.layer = layer

    @property
    def params(self) -> dict:
        return self.layer.get_attributes()

    @property
    def name(self) -> str:
        return self.layer.get_friendly_name()

    @property
    def type(self) -> str:
        return self.layer.get_type_name()

    @property
    def precision(self) -> str:
        return self.layer.get_element_type().get_type_name()

    def get_child_names(self) -> List[str]:
        children = []
        for output in self.layer.outputs():
            children = [i.get_node().get_friendly_name() for i in output.get_target_inputs()]
        return children

    @staticmethod
    def get_blob_size(shape: List[int]) -> int:
        return reduce(operator.mul, shape, 1)

    def get_outputs_number(self) -> int:
        return len(self.layer.outputs())

    def get_output_precision(self, index: int) -> int:
        return self.layer.outputs()[index].get_element_type().get_type_name()

    def get_output_shape(self, index: int) -> list:
        output = self.layer.output(index)
        return get_shape_for_node_safely(output)

    def get_output_blobs_total_size(self) -> int:
        return sum(self.get_blob_size(self.get_output_shape(i)) for i in range(self.get_outputs_number()))

    def get_inputs_number(self) -> int:
        return len(self.layer.inputs())

    def get_input_precision(self, index: int) -> int:
        return self.layer.inputs()[index].get_element_type().get_type_name()

    def get_input_shape(self, index: int) -> list:
        return get_shape_for_node_safely(self.layer.input(index))

    def get_input_blobs_total_size(self) -> int:
        return sum(self.get_blob_size(self.get_input_shape(i)) for i in range(self.get_inputs_number()))

    def get_ops_per_element(self) -> float:
        raise NotImplementedError

    def get_ops(self) -> float:
        return self.get_ops_per_element() * self.get_output_blobs_total_size()

    def get_input_channel(self) -> int:
        input_data = self.layer.in_data[0]
        channels_index = input_data.layout.index('C')
        input_channel = input_data.shape[channels_index]
        return input_channel

    def _get_params(self, indexes: Iterable[int]) -> Dict[str, Tuple[int, int]]:
        result = {}
        for i in indexes:
            source_node = self.layer.input(i).get_source_output().get_node()
            if source_node.get_type_name() == 'FakeQuantize':
                fq_provider = LayerTypesManager.provider(source_node)
                result[source_node.get_friendly_name()] = fq_provider.get_quantized_params()
            else:
                if source_node.get_type_name() in {'Reshape', 'Convert'}:
                    source_node = source_node.input(0).get_source_output().get_node()
                if source_node.get_type_name() == 'Constant':
                    blob = LayerTypesManager.provider(source_node).get_data()
                    result[source_node.get_friendly_name()] = (
                        float(reduce(operator.mul, self.get_input_shape(i), 1)),
                        (blob == 0).sum()
                    )
        return result

    def get_params(self) -> Dict[str, Tuple[int, int]]:
        indexes = range(len(self.layer.inputs()))
        return self._get_params(indexes)

    @staticmethod
    def unpack(data):
        # skip big blobs unpack due to OOM error
        if data.nbytes > 500 * 1024 * 1024:
            return data

        if data.dtype == np.uint16:
            try:
                fp16_data = np.array([struct.unpack('e', struct.pack('H', b)) for b in data.ravel()])
            except struct.error:
                fp16_data = np.array([struct.unpack('f', struct.pack('2H', 0, b)) for b in data.ravel()])
            return fp16_data.astype(np.float32).reshape(data.shape)
        if data.dtype == np.int16:
            try:
                fp16_data = np.array([struct.unpack('e', struct.pack('h', b)) for b in data.ravel()])
            except struct.error:
                fp16_data = np.array([struct.unpack('f', struct.pack('2h', 0, b)) for b in data.ravel()])
            return fp16_data.astype(np.float32).reshape(data.shape)
        return data

    @property
    def div(self) -> int:
        return 1

    @property
    def mul(self) -> int:
        return 1

    @property
    def min(self) -> int:
        return 1

    @property
    def max(self) -> int:
        return 1

    @property
    def add(self) -> int:
        return 1

    @property
    def exp(self) -> int:
        return 1

    @property
    def sum(self) -> int:
        return 1

    @property
    def sub(self) -> int:
        return 1

    @property
    def cmp(self) -> int:
        return 1

    @property
    def sqrt(self) -> int:
        return 1

    @property
    def log(self) -> int:
        return 1

    @property
    def round(self) -> int:
        return 1


class Convolution(LayerType):
    layer_types = ['Convolution']

    @property
    def filter_shape(self) -> List[int]:
        return self.get_input_shape(1)

    @property
    def group(self) -> int:
        return 1

    @property
    def kernel_spatial_size(self) -> int:
        return reduce(operator.mul, list(self.filter_shape)[2:], 1)

    def get_ops_per_element(self) -> float:
        input_shape = self.get_input_shape(0)
        input_channel = input_shape[1]

        # (mul + add) x ROI size
        flops_per_element = (self.mul + self.add) * (input_channel / self.group) * self.kernel_spatial_size
        return flops_per_element


class DeformableConvolution(Convolution):
    layer_types = ['DeformableConvolution']

    @property
    def filter_shape(self) -> List[int]:
        return self.get_input_shape(2)


class GroupConvolution(Convolution):
    layer_types = ['GroupConvolution']
  
    @property
    def group(self) -> int:
        return self.filter_shape[0]

    @property
    def kernel_spatial_size(self) -> int:
        return reduce(operator.mul, list(self.filter_shape)[3:], 1)


class Acosh(LayerType):
    """
    Formula: acosh(x) = ln(x + (x*x - 1)^(1/2))
    """
    layer_types = ['Acosh']

    def get_ops_per_element(self) -> float:
        return self.log + self.sum + self.sub + self.sqrt + self.mul


class Asinh(LayerType):
    """
    Formula: asinh(x) = ln(x + (x*x + 1)^(1/2))
    """
    layer_types = ['Asinh']

    def get_ops_per_element(self) -> float:
        return self.log + 2 * self.sum + self.sqrt + self.mul


class Atanh(LayerType):
    """
    Formula: atanh(x) = 0.5 * ln( (1 + x)/(1 - x) )
    """
    layer_types = ['Atanh']

    def get_ops_per_element(self) -> float:
        return self.mul + self.log + self.sum + self.sub + self.div


class Mish(LayerType):
    """
    Formula: Mish(x) = x*tanh(ln(1 + e^x))
    """
    layer_types = ['Mish']

    def get_ops_per_element(self) -> float:
        return self.mul + Tanh.get_ops_per_element(self) + self.log + self.sum + self.exp


class HSwish(LayerType):
    """
    Formula: HSwish(x) = x / frac{min(max(x + 3, 0), 6)}{6}
    """
    layer_types = ['HSwish']

    def get_ops_per_element(self) -> float:
        return self.div + self.min + self.max + self.sum


class HardSigmoid(LayerType):
    """
    Formula: y(x) = max(0, min(1, alpha * x + beta))
    """
    layer_types = ['HardSigmoid']

    def get_ops_per_element(self) -> float:
        return self.max + self.min + self.mul + self.sum


class SoftPlus(LayerType):
    """
    Formula: SoftPlus(x) = ln(e^x + 1)
    """
    layer_types = ['SoftPlus']

    def get_ops_per_element(self) -> float:
        return self.log + self.exp + self.sum


class Swish(LayerType):
    """
    Formula: Swish(x) = x / (1.0 + 1/(e^(beta * x)))
    """
    layer_types = ['Swish']

    def get_ops_per_element(self) -> float:
        return self.div + self.sum + self.div + self.exp + self.mul


class ReduceL(LayerType):
    layer_types = ['ReduceL1', 'ReduceL2']

    def get_ops_per_element(self) -> float:
        return Normalize.get_ops_per_element(self)


class Deconvolution(LayerType):
    layer_types = ['Deconvolution', 'ConvolutionBackPropData', 'GroupConvolutionBackpropData']

    def get_ops_per_element(self) -> float:

        input_shape = self.get_input_shape(0)
        filter_shape = self.get_input_shape(1)
        input_channel = input_shape[1]
        if self.layer.get_type_name() == 'GroupConvolutionBackpropData':
            group = filter_shape[0]
            kernel_spatial_size = reduce(operator.mul, list(filter_shape)[3:], 1)
        else:
            group = 1
            kernel_spatial_size = reduce(operator.mul, list(filter_shape)[2:], 1)
        stride_spatial_size = reduce(operator.mul, _get_param_values(self.params, 'strides', 'stride'), 1)
        flops_per_element = (self.mul + self.add) * (input_channel / group) * kernel_spatial_size / stride_spatial_size
        return flops_per_element


class Relu(LayerType):
    """
    Formula: cmp + mul
    """
    layer_types = ['ReLu']

    def get_ops_per_element(self) -> int:
        return self.cmp + self.mul


class Normalize(LayerType):
    """
    Formula: cmp + mul + sum + sqrt
    """
    layer_types = ['Normalize', 'NormalizeL2']

    def get_ops_per_element(self) -> int:
        return self.cmp + self.mul + self.sum + self.sqrt


class Norm(LayerType):
    """
    Formula: (mul + add) x ROI size + div
    """
    layer_types = ['Norm']

    def get_ops_per_element(self) -> int:
        roi_size = 2 * int(self.layer.params['local-size'])
        if self.params['region'] == 'across':
            roi_size = roi_size * int(self.params['local-size'])
        flops_per_element = (self.mul + self.add) * roi_size + self.div
        return flops_per_element


class Pooling(LayerType):
    """
    Formula: (max or add) x ROI size
    """
    layer_types = ['Pooling', 'MaxPool', 'AvgPool']

    def get_ops_per_element(self) -> int:
        kernel_spatial_size = reduce(operator.mul, _get_param_values(self.params, 'kernel', 'kernel'), 1)
        flops_per_element = self.add * kernel_spatial_size
        return flops_per_element


class FullyConnected(LayerType):
    layer_types = ['FullyConnected']

    def get_ops_per_element(self) -> int:
        input_size = reduce(operator.mul, self.layer.in_data[0].shape[1:], 1)
        flops_per_element = 2 * input_size
        return flops_per_element


class SoftMax(LayerType):
    """
    Formula: max + sub + exp + sum + div
    """
    layer_types = ['Softmax']

    def get_ops_per_element(self) -> int:
        return self.max + self.sub + self.exp + self.sum + self.div


class LogSoftMax(SoftMax):
    layer_types = ['LogSoftMax']

    def get_ops_per_element(self) -> int:
        return super().get_ops_per_element() + self.log


class Elu(LayerType):
    layer_types = ['Elu']

    def get_ops_per_element(self) -> int:
        return 3


class Eltwise(LayerType):
    layer_types = ['Eltwise']

    def get_ops_per_element(self) -> int:
        input_blob_count = self.get_inputs_number()
        flops_per_element = 2 * input_blob_count - 1
        return flops_per_element


class ScaleShift(LayerType):
    layer_types = ['ScaleShift']

    def get_ops_per_element(self) -> int:
        return 2


class BatchNormalization(ScaleShift):
    layer_types = ['BatchNormalization']


class Power(LayerType):
    """
    Formula: mul + add + (power - 1) x mul
    """
    layer_types = ['Power']

    def get_ops_per_element(self) -> float:
        power_op = self.layer.input(1).get_source_output().get_node()
        if power_op.get_type_name() == 'Constant':
            provider = LayerTypesManager.provider(power_op)
            if provider.get_output_blobs_total_size == 1:
                power = float(provider.get_data())
            else:
                # Different number of ops per element
                raise NotImplementedError
        else:
            raise NotImplementedError
        flops_per_element = self.mul + self.add + (power - 1) * self.mul
        return flops_per_element


class Clamp(LayerType):
    layer_types = ['Clamp']

    def get_ops_per_element(self) -> int:
        return self.min + self.max


class PsRoiPooling(LayerType):
    layer_types = ['PSROIPooling']

    def get_ops_per_element(self) -> int:
        in_dims = list(self.layer.inputs()[0].get_shape())
        out_dims = list(self.layer.outputs()[0].get_shape())
        # real kernel sizes are read from input, so approximation is used
        size = 3 if len(in_dims) == 5 else 2
        kernel_spatial_size = 1
        for i in range(0, size):
            kernel_spatial_size *= in_dims[-1 - i] // out_dims[-1 - i]
        flops_per_element = 1 * kernel_spatial_size
        return flops_per_element


class RoiPooling(PsRoiPooling):
    layer_types = ['ROIPooling']


class Mvn(LayerType):
    layer_types = ['MVN']

    def get_ops_per_element(self) -> int:
        flops_per_element = 5 if self.params['normalize_variance'] == '1' else 2
        return flops_per_element


class Grn(LayerType):
    layer_types = ['GRN']

    def get_ops_per_element(self) -> int:
        return 3


class Prelu(LayerType):
    layer_types = ['PReLU']

    def get_ops_per_element(self) -> int:
        return 2


class ArgMax(LayerType):
    layer_types = ['ArgMax']

    def get_ops_per_element(self) -> int:
        top_k = int(self.params['top_k'])
        axis_index = int(self.layer.params['axis']) if 'axis' in self.params.keys() else 0
        in_dims = self.layer.in_data[0].shape
        roi_size = in_dims[axis_index] if axis_index != 0 else self.get_input_blobs_total_size()
        flops_per_element = 1 * (roi_size * top_k - top_k * (top_k + 1) / 2)  # cmp x ROI size
        return flops_per_element


class Interp(LayerType):
    layer_types = ['Interp']

    def get_ops_per_element(self) -> int:
        return 9


class Round(LayerType):
    layer_types = ['Round']

    def get_ops_per_element(self) -> int:
        return self.round


class Sigmoid(LayerType):
    layer_types = ['Sigmoid']

    def get_ops_per_element(self) -> int:
        return 3


class Gemm(LayerType):
    layer_types = ['GEMM', 'MatMul']

    def get_ops_per_element(self) -> int:
        in_dims = list(self.layer.inputs()[0].get_shape())
        flops_per_element = 2 * in_dims[-1]
        return flops_per_element


class Tanh(LayerType):
    layer_types = ['Tanh']

    def get_ops_per_element(self) -> int:
        return Sigmoid.get_ops_per_element(self) + 2 * self.mul + self.sum


class Add(LayerType):
    layer_types = ['Add']

    def get_ops_per_element(self) -> int:
        return self.add


class Subtract(LayerType):
    layer_types = ['Subtract']

    def get_ops_per_element(self) -> int:
        return self.div


class Multiply(LayerType):
    layer_types = ['Multiply']

    def get_ops_per_element(self) -> int:
        return self.mul


class Divide(LayerType):
    layer_types = ['Divide']

    def get_ops_per_element(self) -> int:
        return self.div


class Less(LayerType):
    layer_types = ['Less', 'LessEqual']

    def get_ops_per_element(self) -> int:
        return self.cmp

    def get_params(self) -> Dict[str, Tuple[int, int]]:
        return None


class Greater(LayerType):
    layer_types = ['Greater ', 'GreaterEqual']

    def get_ops_per_element(self) -> int:
        return self.cmp

    def get_params(self) -> Dict[str, Tuple[int, int]]:
        return None


class Exp(LayerType):
    layer_types = ['Exp']

    def get_ops_per_element(self) -> int:
        return self.exp

    def get_params(self) -> Dict[str, Tuple[int, int]]:
        return super()._get_params([0])


class Log(LayerType):
    layer_types = ['Log']

    def get_ops_per_element(self) -> int:
        return self.log


class NonMathLayer(LayerType):
    def get_ops_per_element(self) -> int:
        return 0


class Concat(NonMathLayer):
    layer_types = ['Concat']


class Constant(NonMathLayer):
    layer_types = ['Const', 'Constant']

    def get_params(self) -> Dict[str, Tuple[int, int]]:
        return None

    def get_data(self):
        data = self.layer.get_data()
        return LayerType.unpack(data)


class Crop(NonMathLayer):
    layer_types = ['Crop']


class Gather(NonMathLayer):
    """Data movement operations"""
    layer_types = ['Gather', 'GatherND', 'GatherTree', 'GatherElements']

    def get_params(self) -> Dict[str, Tuple[int, int]]:
        return super()._get_params([0])


class NonMaxSuppression(NonMathLayer):
    layer_types = ['NonMaxSuppression']


class Parameter(NonMathLayer):
    layer_types = ['Input', 'Parameter']

    def get_inputs_number(self):
        return 0

    def get_params(self) -> Dict[str, Tuple[int, int]]:
        return None


class Range(NonMathLayer):
    layer_types = ['Range']


class Result(NonMathLayer):
    layer_types = ['Output', 'Result']

    def get_outputs_number(self):
        return 0

    def get_params(self) -> Dict[str, Tuple[int, int]]:
        return None


class Reshape(NonMathLayer):
    layer_types = ['Reshape']

    def get_params(self) -> Dict[str, Tuple[int, int]]:
        return super()._get_params([0])


class ReduceMin(NonMathLayer):
    layer_types = ['ReduceMin']


class ReduceMax(NonMathLayer):
    layer_types = ['ReduceMax']


class ReverseSequence(NonMathLayer):
    layer_types = ['ReverseSequence']


class Squeeze(NonMathLayer):
    layer_types = ['Squeeze']

    def get_params(self) -> Dict[str, Tuple[int, int]]:
        return super()._get_params([0])


class Select(NonMathLayer):
    layer_types = ['Select']


class SparseToDense(NonMathLayer):
    layer_types = ['SparseToDense']


class ScatterNDUpdate(NonMathLayer):
    layer_types = ['ScatterNDUpdate']


class Tile(NonMathLayer):
    layer_types = ['Tile']


class Unsqueeze(NonMathLayer):
    layer_types = ['Unsqueeze']

    def get_params(self) -> Dict[str, Tuple[int, int]]:
        return super()._get_params([0])


class Pad(NonMathLayer):
    layer_types = ['Pad']

    def get_params(self) -> Dict[str, Tuple[int, int]]:
        return None


class OneHot(NonMathLayer):
    layer_types = ['OneHot']

    def get_params(self) -> Dict[str, Tuple[int, int]]:
        return None


class Split(NonMathLayer):
    layer_types = ['Split']

    def get_params(self) -> Dict[str, Tuple[int, int]]:
        return self._get_params([0])


class StridedSlice(NonMathLayer):
    layer_types = ['StridedSlice']

    def get_params(self) -> Dict[str, Tuple[int, int]]:
        return super()._get_params([0])


class Permute(NonMathLayer):
    layer_types = ['Permute']

    def get_params(self) -> Dict[str, Tuple[int, int]]:
        return None


class Priorbox(NonMathLayer):
    layer_types = ['Priorbox', 'PriorboxClustered']


class Proposal(NonMathLayer):
    layer_types = ['Proposal']


class Loop(NonMathLayer):
    layer_types = ['Loop']
