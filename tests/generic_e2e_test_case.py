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

import math
import os
import shutil
from pathlib import Path
from typing import Tuple

from model_analyzer import main
from tests.constants import MODELS_PATH
from tests.utils import TestNamespace

MODEL_PATHS = Tuple[Path, Path]
MODEL_PATHS_TYPE = Tuple[str, str, str]


class GenericE2ETestCase:
    data_dir = MODELS_PATH
    artifacts_dir = None

    def get_model_paths(self, xml_path: Path, bin_path: Path) -> tuple:
        xml_path = self.data_dir / xml_path
        bin_path = (self.data_dir / bin_path) if bin_path else None
        return xml_path, bin_path

    def custom_set_up(self, xml_path: Path, bin_path: Path = None):
        xml_path, bin_path = self.get_model_paths(xml_path, bin_path)
        shutil.copy(xml_path, self.artifacts_dir)

        report_name = 'model_report.csv'
        report_path = os.path.join(self.artifacts_dir, report_name)
        namespace = TestNamespace(model=xml_path,
                                  weights=bin_path,
                                  report_dir=self.artifacts_dir,
                                  per_layer_mode=False,
                                  per_layer_report=None,
                                  model_report=report_name,
                                  sparsity_ignored_layers='',
                                  sparsity_ignore_first_conv=False,
                                  sparsity_ignore_fc=False,
                                  ignore_unknown_layers=True)
        main(namespace)

        return report_path, xml_path

    @staticmethod
    def custom_tear_down(report_path):
        os.remove(report_path)

    @staticmethod
    def read_report(report_path):
        with open(report_path) as file:
            first = [i.strip() for i in file.readline().split(',')]
            second_line = file.readline().split(',')
            second = []
            for value in second_line:
                try:
                    second.append(float(value))
                except ValueError:
                    second.append(value)
            res = {}
            for key, value in zip(first, second):
                res[key] = value
        return res

    def find_expected_by_name(self, xml_path):
        res = list(filter(lambda l: l['xml_path'] == xml_path, self.data))
        return res[0]['analysis_report'] if res else None

    @staticmethod
    def compare_float_dictionaries(actual, expected):
        for key, value in expected.items():
            assert math.isclose(value, actual[key], rel_tol=1e-02), f"Expected {key} value {value}, received {actual[key]}"
