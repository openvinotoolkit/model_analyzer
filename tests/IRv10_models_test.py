"""
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

import pytest

from model_analyzer.model_metadata import ModelMetaData, ModelTypes
from tests.generic_e2e_test_case import GenericE2ETestCase, MODEL_PATHS, MODEL_PATHS_TYPE
from tests.utils import load_test_config

ARTIFACTS_DIR, DATA = load_test_config('IRv10_models.json')


class TestCaseR1Models(GenericE2ETestCase):
    @classmethod
    def setup_class(cls):
        cls.artifacts_dir = ARTIFACTS_DIR
        cls.data = DATA
        cls.VERSION = '2020_r1'

    @pytest.fixture(params=[(model['xml_path'], model['bin_path']) for model in DATA])
    def get_model_info(self, request) -> MODEL_PATHS:
        return request.param

    @pytest.fixture(params=[(model['xml_path'], model['bin_path'], model['type']) for model in DATA])
    def get_model_info_topology(self, request) -> tuple:
        return request.param

    def test_analysis_report(self, get_model_info: MODEL_PATHS):
        xml_path, bin_path = get_model_info
        report_path, _ = self.custom_set_up(xml_path, bin_path)
        res = self.read_report(report_path)
        expected = self.find_expected_by_name(xml_path)
        self.compare_float_dictionaries(res, expected)
        self.custom_tear_down(report_path)

    def test_guess_topology_type(self, get_model_info_topology: MODEL_PATHS_TYPE):
        xml_path, bin_path, model_type = get_model_info_topology
        if 'road-segmentation-adas-0001' in xml_path:
            return 
        cannot_recognize = ['face_recognition', 'object_attributes', 'optical_character_recognition',
                            'head_pose_estimation', 'human_pose_estimation', 'image_processing', 'feature_extraction',
                            'action_recognition', 'detection-']

        if model_type not in cannot_recognize:
            xml_path = self.data_dir / xml_path
            bin_path = self.data_dir / bin_path

            if model_type == 'detection':
                expected = ModelTypes.SSD
            elif model_type == 'instance_segmentation':
                expected = ModelTypes.INSTANCE_SEGM
            elif model_type == 'semantic_segmentation':
                expected = ModelTypes.SEMANTIC_SEGM
            elif model_type == 'yolo':
                expected = ModelTypes.YOLO
            else:
                expected = model_type

            result = ModelMetaData(xml_path, bin_path).guess_topology_type()

            assert expected == result
