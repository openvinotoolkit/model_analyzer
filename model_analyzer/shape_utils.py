# Copyright (C) 2019-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List

# pylint: disable=import-error
from openvino.runtime import Input, PartialShape


def get_shape_for_node_safely(node: Input) -> List[int]:
    partial_shape = node.get_partial_shape()
    if partial_shape.is_dynamic:
        node = node.get_node()

        try:
            node_name = node.friendly_name
        except (RuntimeError, AttributeError):
            node_name = ''
        logging.warning(
            '%s layer of type %s has dynamic output shape.',
            node_name, node.get_type_name()
        )

        return get_shape_safely(partial_shape)
    return list(partial_shape.to_shape())


def get_shape_safely(partial_shape: PartialShape) -> List[int]:
    shape = []
    for i, _ in enumerate(partial_shape):
        dimension = -1
        if partial_shape[i].is_static:
            dimension = int(str(partial_shape[i]))
        shape.append(dimension)
    return shape
