"""
 Model Analyzer

 Copyright (c) 2021 Intel Corporation

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
from typing import List, Union

# pylint: disable=import-error
from openvino.runtime import Output, Input, PartialShape


def get_shape_for_node_safely(io_node: Union[Input, Output]) -> List[int]:
    partial_shape = io_node.get_partial_shape()
    if partial_shape.is_dynamic:
        node = io_node.get_node()

        try:
            node_name = io_node.any_name
        except RuntimeError:
            node_name = ''
        logging.warning(
            '%s layer of type %s has dynamic output shape.',
            node_name, node.get_type_name()
        )

        return get_shape_safely(partial_shape)
    return [s for s in partial_shape.to_shape()]


def get_shape_safely(partial_shape: PartialShape) -> List[int]:
    shape = []
    for i, _ in enumerate(partial_shape):
        dimension = -1
        if partial_shape[i].is_static:
            dimension = int(str(partial_shape[i]))
        shape.append(dimension)
    return shape
