# Copyright (C) 2019-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import operator
from functools import reduce
from typing import Tuple, Dict

import numpy as np

# pylint: disable=no-name-in-module,import-error
from openvino.runtime import Node

from model_analyzer.layer_provider import LayerTypesManager, LayerType


# Now we don't have a method of counting operations of these layers
# pylint: disable=abstract-method
class Abs(LayerType):
    layer_types = ['Abs']


class BatchToSpace(LayerType):
    layer_types = ['BatchToSpace']

    def get_params(self) -> Dict[str, Tuple[int, int]]:
        return None


class Broadcast(LayerType):
    layer_types = ['Broadcast']

    def get_params(self) -> Dict[str, Tuple[int, int]]:
        return super()._get_params([0])


class Bucketize(LayerType):
    layer_types = ['Bucketize']


class Convert(LayerType):
    layer_types = ['Convert']

    def get_params(self) -> Dict[str, Tuple[int, int]]:
        return None


class CtcGreedyDecoder(LayerType):
    layer_types = ['CtcGreedyDecoder']


class CtcLoss(LayerType):
    layer_types = ['CtcLoss']


class DetectionOutput(LayerType):
    layer_types = ['DetectionOutput']

    def get_params(self) -> Dict[str, Tuple[int, int]]:
        return None


class Erf(LayerType):
    layer_type = ['Erf']


class ExperimentalDetectronDetectionOutput(LayerType):
    layer_types = ['ExperimentalDetectronDetectionOutput']


class ExperimentalDetectronGenerateProposalsSingleImage(LayerType):
    layer_types = ['ExperimentalDetectronGenerateProposalsSingleImage']


class ExperimentalDetectronPriorGridGenerator(LayerType):
    layer_types = ['ExperimentalDetectronPriorGridGenerator']


class ExperimentalDetectronRoiFeatureExtractor(LayerType):
    layer_types = ['ExperimentalDetectronRoiFeatureExtractor']


class ExperimentalDetectronTopkRois(LayerType):
    layer_types = ['ExperimentalDetectronTopkRois']


class ExperimentalSparseWeightedSum(LayerType):
    layer_types = ['ExperimentalSparseWeightedSum']


class Flatten(LayerType):
    layer_types = ['Flatten']


class Interpolate(LayerType):
    layer_types = ['Interpolate']


class LSTMSequence(LayerType):
    layer_types = ['LSTMSequence']


class PredictionHeatMap(LayerType):
    layer_types = ['PredictionHeatMap']


class RegionYolo(LayerType):
    layer_types = ['RegionYolo']


class ReorgYolo(LayerType):
    layer_types = ['ReorgYolo']


class Resample(LayerType):
    layer_types = ['ReSample']


class TensorIterator(LayerType):
    layer_types = ['TensorIterator']

    def get_params(self) -> Tuple[int, int]:
        return None


class TopK(LayerType):
    layer_types = ['TopK']


class Transpose(LayerType):
    layer_types = ['Transpose']

    def get_params(self) -> Dict[str, Tuple[int, int]]:
        return super()._get_params([0])


class VariadicSplit(LayerType):
    layer_types = ['VariadicSplit']

    def get_params(self) -> Dict[str, Tuple[int, int]]:
        return self._get_params([0])


class SpaceToBatch(LayerType):
    layer_types = ['SpaceToBatch']

    def get_params(self) -> Dict[str, Tuple[int, int]]:
        return None


class SpatialTransformer(LayerType):
    layer_types = ['SpatialTransformer']


class Slice(LayerType):
    layer_types = ['Slice']


class FakeQuantize(LayerType):
    layer_types = ['FakeQuantize']

    def __init__(self, layer: Node):
        super().__init__(layer)
        limits = []
        for i in range(1, self.get_inputs_number()):
            limits.append(LayerTypesManager.provider(self.layer.input(i).get_source_output().get_node()).get_data())
        # pylint: disable=W0632
        self.input_low, self.input_high, self.output_low, self.output_high = limits
        self.levels = int(self.params['levels'])

    def get_params(self) -> Dict[str, Tuple[int, int]]:
        return None

    def get_quantized_params(self) -> Tuple[int, int]:
        source_input = self.layer.input(0).get_source_output().get_node()
        if source_input.get_type_name() == 'Convert':
            source_input = source_input.input(0).get_source_output().get_node()
        if source_input.get_type_name() != 'Constant':
            return 0, 0
        data_input = LayerTypesManager.provider(source_input).get_data()

        shape = data_input.shape
        input_low = np.broadcast_to(self.input_low, shape)
        input_high = np.broadcast_to(self.input_high, shape)
        output_low = np.broadcast_to(self.output_low, shape)
        output_high = np.broadcast_to(self.output_high, shape)
        params = reduce(operator.mul, shape, 1)
        # pylint: disable=E1101
        eps = np.finfo(data_input.dtype).tiny
        value = np.round((data_input - input_low) / (input_high - input_low) * (self.levels - 1))
        zero_point = np.round(output_low / (output_high - output_low + eps) * (self.levels - 1))
        zeros = (value + zero_point == 0).sum()
        return params, zeros
