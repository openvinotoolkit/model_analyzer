# Copyright (C) 2019-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
from typing import List

# pylint: disable=import-error
from openvino.runtime import Node, Layout

from model_analyzer.constants import LayoutTypes


def is_image_info_layout(layout: List[str]) -> bool:
    return layout in (LayoutTypes.NC, LayoutTypes.CN)


def is_batched_image_layout(layout: List[str]) -> bool:
    return is_image_layout(layout) and 'N' in layout


def is_image_layout(layout: List[str]) -> bool:
    return (
            'H' in layout and
            'W' in layout and
            'C' in layout
    )


def get_fully_undefined_node_layout(node: Node) -> List[str]:
    return ['?'] * len(node.shape)


def parse_node_layout(node: Node) -> List[str]:
    layout: Layout = node.layout
    if layout.empty:
        return get_fully_undefined_node_layout(node)

    layout_match = re.search(r'\[(?P<layout>.*)]', str(layout))
    if not layout_match:
        return get_fully_undefined_node_layout(node)

    clear_layout = layout_match.group('layout')
    return [dim for dim in clear_layout.split(',')]
