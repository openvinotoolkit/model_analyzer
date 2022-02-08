# Copyright (C) 2019-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json

import shutil
from pathlib import Path


def load_test_config(config_name):
    tests_dir = Path(__file__).resolve().parent

    data_dir = tests_dir / 'data'
    config_path = data_dir / config_name
    artifacts_dir = tests_dir / 'output'

    if artifacts_dir.is_dir():
        shutil.rmtree(artifacts_dir)
    artifacts_dir.mkdir(exist_ok=True)

    with config_path.open() as file:
        data = json.load(file)
    for model_data in data:
        if 'xml_path' in model_data:
            model_data['xml_path'] = Path(model_data['xml_path'])
        if 'bin_path' in model_data:
            model_data['bin_path'] = Path(model_data['bin_path'])
        if 'path' in model_data:
            model_data['path'] = Path(model_data['path'])
    return artifacts_dir, data


class TestNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
