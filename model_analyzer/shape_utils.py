import logging
from typing import List, Union

from openvino.runtime import Output, Input, PartialShape


def get_shape_for_node_safely(node: Union[Input, Output]) -> List[int]:
    partial_shape = node.get_partial_shape()
    if partial_shape.is_dynamic:
        node = node.get_node()
        logging.warning('%s layer of type %s has dynamic output shape.', node.get_friendly_name(), node.get_type_name())
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
