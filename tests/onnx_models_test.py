# Copyright (C) 2019-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from tests.utils import load_test_config

from tests.generic_e2e_test_case import GenericE2ETestCase

ARTIFACTS_DIR, DATA = load_test_config('onnx_models.json')


class TestOnnxModelsCase(GenericE2ETestCase):
    VERSION = 'onnx'

    @classmethod
    def setup_class(cls):
        cls.artifacts_dir = Path(ARTIFACTS_DIR)
        cls.data = DATA
        cls.VERSION = 'onnx'

    @pytest.fixture(scope="function", params=[i['model_path'] for i in DATA])
    def get_model_path(self, request) -> Path:
        return request.param

    def test_analysis_report(self, get_model_path: Path):
        model_path = get_model_path
        report_path, _ = self.custom_set_up(model_path)
        res = self.read_report(report_path)
        found_model = list(filter(lambda l: l['model_path'] == model_path, self.data))
        expected = found_model[0]['analysis_report'] if found_model else None
        self.compare_float_dictionaries(res, expected)
        self.custom_tear_down(report_path)
