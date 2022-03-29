# Copyright (C) 2019-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Optional, Tuple, Dict

import pytest
import os

from model_analyzer.model_metadata import ModelMetaData
from tests.utils import load_test_config

_, CONFIG_DATA = load_test_config('IRv10_models.json')


def get_xml_and_bin_path(model_name: str) -> Tuple[Path, Path]:
    model_dir = Path(os.environ['MODELS_PATH'])
    model_data = next(filter(lambda model_info: model_info['name'] == model_name, CONFIG_DATA))
    model_xml_path = model_dir / model_data['xml_path']
    model_bin_path = model_dir / model_data['bin_path']
    return model_xml_path, model_bin_path


@pytest.fixture(params=[
    ('gaze-estimation-adas-0002', False),
])
def is_obsolete_model_test_params(request) -> Tuple[str, bool]:
    return request.param


def test_is_obsolete(is_obsolete_model_test_params: Tuple[str, bool]):
    model_name, expected = is_obsolete_model_test_params
    mmd = ModelMetaData(*get_xml_and_bin_path(model_name))
    result = mmd.is_obsolete()
    assert result == expected


@pytest.fixture(params=[
    ('person-vehicle-bike-detection-crossroad-1016', 10),
])
def model_version_test_params(request) -> Tuple[str, int]:
    return request.param


def test_get_ir_version(model_version_test_params: Tuple[str, int]):
    model, expected = model_version_test_params
    nmd = ModelMetaData(*get_xml_and_bin_path(model))
    result = nmd.get_ir_version()
    assert result == expected


@pytest.fixture(params=[
    ('text-detection-0004', {
        'caffe_parser_path': 'DIR',
        'data_type': 'FP32',
        'disable_nhwc_to_nchw': 'False',
        'disable_omitting_optional': 'False',
        'disable_resnet_optimization': 'False',
        'disable_weights_compression': 'False',
        'enable_concat_optimization': 'False',
        'enable_flattening_nested_params': 'False',
        'enable_ssd_gluoncv': 'False',
        'extensions': 'DIR',
        'framework': 'tf',
        'freeze_placeholder_with_value': '{}',
        'generate_deprecated_IR_V7': 'False',
        'input': 'Placeholder',
        'input_model': 'DIR/pixel_link_mobilenet_v2.pb',
        'input_model_is_text': 'False',
        'input_shape': '[1,768,1280,3]',
        'k': 'DIR/CustomLayersMapping.xml',
        'keep_shape_ops': 'True',
        'legacy_ir_generation': 'False',
        'legacy_mxnet_model': 'False',
        'log_level': 'ERROR',
        'mean_scale_values': "{'Placeholder': {'mean': array([127.5, 127.5, 127.5]), "
                             "'scale': array([127.5])}}",
        'mean_values': 'Placeholder[127.5,127.5,127.5]',
        'model_name': 'text-detection-0004',
        'output': "['model/segm_logits/add', 'model/link_logits_/add']",
        'output_dir': 'DIR',
        'placeholder_data_types': '{}',
        'placeholder_shapes': "{'Placeholder': array([   1,  768, 1280,    3])}",
        'progress': 'False',
        'remove_memory': 'False',
        'remove_output_softmax': 'False',
        'reverse_input_channels': 'True',
        'save_params_from_nd': 'False',
        'scale_values': 'Placeholder[127.5]',
        'silent': 'False',
        'static_shape': 'False',
        'stream_output': 'False',
        'transform': '',
        'version': '2021.4.0-3827-c5b65f2cb1d-releases/2021/4'
    })
])
def mo_parameters_test_params(request) -> Tuple[str, Dict[str, str]]:
    return request.param


def test_get_mo_params(mo_parameters_test_params: Tuple[str, Dict[str, str]]):
    model_name, expected = mo_parameters_test_params
    mmd = ModelMetaData(*get_xml_and_bin_path(model_name))
    result = mmd.get_mo_params()
    assert result == expected


@pytest.fixture(params=[
    ('handwritten-score-recognition-0003', ('Convolution', 'LayerThatDoesNotExist'), True),
    ('head-pose-estimation-adas-0001', ('FakeQuantize',), False)
])
def has_layer_type_test_params(request) -> Tuple[str, Tuple[str,...], bool]:
    return request.param


def test_has_layer_type(has_layer_type_test_params: Tuple[str, Tuple[str,...], bool]):
    model_name, layer_types, expected = has_layer_type_test_params
    mmd = ModelMetaData(*get_xml_and_bin_path(model_name))
    result = mmd.has_op_of_type(*layer_types)
    assert result == expected


@pytest.fixture(params=[
    ('yolo-v2-ava-0001', 20),
])
def num_classes_test_params(request) -> Tuple[str, int]:
    return request.param


def test_get_num_classes(num_classes_test_params: Tuple[str, int]):
    model_name, expected = num_classes_test_params
    mmd = ModelMetaData(*get_xml_and_bin_path(model_name))
    result = mmd.get_num_classes()
    assert result == expected


@pytest.fixture(params=[
    ('yolo-v2-ava-0001', None),
])
def background_class_test_params(request) -> Tuple[str, bool]:
    return request.param


def test_has_background_class(background_class_test_params: Tuple[str, Optional[bool]]):
    model_name, expected = background_class_test_params
    mmd = ModelMetaData(*get_xml_and_bin_path(model_name))
    result = mmd.has_background_class()
    assert result == expected


@pytest.fixture(params=[
    ('yolo-v2-ava-0001', False)
])
def is_winograd_test_params(request) -> Tuple[str, bool]:
    return request.param


def test_is_winograd(is_winograd_test_params: Tuple[str, bool]):
    model_name, expected = is_winograd_test_params
    mmd = ModelMetaData(*get_xml_and_bin_path(model_name))
    result = mmd.is_winograd()
    assert result == expected
