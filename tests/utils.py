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
