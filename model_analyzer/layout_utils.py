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
import re
from enum import Enum
from typing import List

# pylint: disable=import-error
from openvino.runtime import Node, Layout

from model_analyzer.constants import LayoutTypes


def is_image_info_layout(layout: Layout) -> bool:
    return layout == LayoutTypes.NC or layout == LayoutTypes.CN


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
