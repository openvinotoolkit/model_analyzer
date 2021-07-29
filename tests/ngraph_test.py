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

import functools
import operator

import ngraph as ng
import numpy as np
import pytest

from model_analyzer.layer_provider import LayerTypesManager


def spatial_shape(shape):
    return functools.reduce(operator.mul, shape, 1)


convolution_data = list()
groupconvolution_data = list()
matmul_data = list()

for b in [1, 32, 128]:
    convolution_data.extend([
        # 2D Convolutions
        (b, 3, [224] * 2, 64, [7] * 2, [2] * 2, [1] * 2, [3] * 2, [3] * 2, 0),
        (b, 128, [28] * 2, 512, [5] * 2, [1] * 2, [1] * 2, [0, 0], [0, 0], 30),
        (b, 512, [7] * 2, 512, [3] * 2, [1] * 2, [1] * 2, [1] * 2, [1] * 2, 15),
        (b, 128, [17] * 2, 128, [1, 7], [1] * 2, [1] * 2, [0, 3], [0, 3], 20),
        (b, 128, [17] * 2, 128, [7, 1], [1] * 2, [1] * 2, [3, 0], [3, 0], 90),
        (b, 19, [128] * 2, 19, [3] * 2, [1] * 2, [64] * 2, [64] * 2, [64] * 2, 5),
        # 3D Convolutions
        (b, 3, [16, 112, 112], 64, [7] * 3, [1, 2, 2], [1] * 3, [1] * 3, [1] * 3, 1),
        (b, 128, [4, 14, 14], 512, [1] * 3, [1] * 3, [1] * 3, [0] * 3, [0] * 3, 10),
    ])
    groupconvolution_data.extend([
        # 3D Convolutions
        (b, 32, [16, 56, 56], 8, 4, [3] * 3, [1] * 3, [1] * 3, [1] * 3, [1] * 3, 0),
        (b, 512, [2, 7, 7], 8, 64, [3] * 3, [1] * 3, [1] * 3, [1] * 3, [1] * 3, 0),
    ])


@pytest.mark.parametrize("batch_size,in_c,in_dims,out_c,kernels,strides,dilations,pads_begin,pads_end,sparsity",
                         convolution_data)
def test_convolution(batch_size, in_c, in_dims, out_c, kernels, strides, dilations, pads_begin, pads_end, sparsity):
    data_shape = [batch_size, in_c, *in_dims]
    filters_shape = [out_c, in_c, *kernels]
    filters_value = np.arange(start=1.0, stop=spatial_shape(filters_shape) + 1.0, dtype=np.float32)
    zeros = int(spatial_shape(filters_shape) * sparsity / 100)
    filters_value[:zeros] = 0
    filters_value = filters_value.reshape(filters_shape)
    param = ng.parameter(data_shape, dtype=np.float32)
    param_filter = ng.constant(filters_value, dtype=np.float32)
    conv = ng.convolution(param, param_filter, strides, pads_begin, pads_end, dilations)
    layer_provider = LayerTypesManager.provider(conv)
    flops_per_element = 2 * in_c * spatial_shape(kernels)
    gops = flops_per_element * spatial_shape(list(conv.outputs()[0].get_partial_shape().to_shape()))

    assert layer_provider.get_ops_per_element() == flops_per_element
    assert layer_provider.get_ops() == gops

    received_params = functools.reduce(operator.add, [x[0] for x in layer_provider.get_params().values()], 0)
    received_zeros = functools.reduce(operator.add, [x[1] for x in layer_provider.get_params().values()], 0)

    assert received_params == spatial_shape(filters_shape)
    assert received_zeros == zeros


@pytest.mark.parametrize("batch_size,in_c,in_dims,out_c,group,kernels,strides,dilations,pads_begin,pads_end,sparsity",
                         groupconvolution_data)
def test_groupconvolution(batch_size, in_c, in_dims, out_c, group, kernels, strides, dilations, pads_begin, pads_end,
                          sparsity):
    data_shape = [batch_size, group * in_c, *in_dims]
    filters_shape = [group, out_c, in_c, *kernels]
    filters_value = np.arange(start=1.0, stop=spatial_shape(filters_shape) + 1.0, dtype=np.float32)
    zeros = int(spatial_shape(filters_shape) * sparsity / 100)
    filters_value[:zeros] = 0
    filters_value = filters_value.reshape(filters_shape)
    param = ng.parameter(data_shape, dtype=np.float32)
    param_filter = ng.constant(filters_value, dtype=np.float32)
    conv = ng.group_convolution(param, param_filter, strides, pads_begin, pads_end, dilations)
    layer_provider = LayerTypesManager.provider(conv)
    flops_per_element = 2 * in_c * spatial_shape(kernels)
    gops = flops_per_element * spatial_shape(list(conv.outputs()[0].get_partial_shape().to_shape()))

    assert layer_provider.get_ops_per_element() == flops_per_element
    assert layer_provider.get_ops() == gops

    received_params = functools.reduce(operator.add, [x[0] for x in layer_provider.get_params().values()], 0)
    received_zeros = functools.reduce(operator.add, [x[1] for x in layer_provider.get_params().values()], 0)

    assert received_params == spatial_shape(filters_shape)
    assert received_zeros == zeros
